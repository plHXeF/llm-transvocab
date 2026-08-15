import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from domain import Card
from llm_service import (
    DEFAULT_STRUCTURED_OUTPUT_TOKENS,
    EVALUATION_PROMPT_VERSION,
    LLMConfigurationError,
    LLMResponseValidationError,
    LLMService,
    LLMServiceError,
    LLMSettings,
    ReasoningEffort,
    RETRY_OUTPUT_TOKEN_INCREMENT,
    SENTENCE_PROMPT_VERSION,
    STRUCTURED_RESPONSE_MAX_RETRIES,
    VOCAB_IMPORT_PROMPT_VERSION,
    VOCAB_IMPORT_OUTPUT_TOKENS,
    VOCAB_IMPORT_RETRY_TOKEN_INCREMENT,
    reasoning_request_options,
)


def completion(
    content,
    *,
    finish_reason=None,
    completion_tokens=None,
    reasoning_tokens=None,
    request_id=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
        ),
        _request_id=request_id,
    )


def mock_client(*contents):
    client = Mock()
    client.chat.completions.create.side_effect = [completion(item) for item in contents]
    return client


class ReasoningMappingTests(unittest.TestCase):
    def test_standard_openai_compatible_effort_mapping(self):
        self.assertIsNone(reasoning_request_options("auto").reasoning_effort)
        self.assertEqual(reasoning_request_options("disabled").reasoning_effort, "none")
        self.assertEqual(reasoning_request_options("minimal").reasoning_effort, "minimal")
        self.assertEqual(reasoning_request_options("medium").reasoning_effort, "medium")
        self.assertEqual(reasoning_request_options("max").reasoning_effort, "max")
        self.assertTrue(reasoning_request_options("low").omit_temperature)

    def test_settings_hide_key_and_have_safe_fingerprint(self):
        secret = "sk-super-secret-value"
        settings = LLMSettings(
            base_url="https://gateway.example/v1",
            model="model-a",
            api_key=secret,
        )
        self.assertNotIn(secret, repr(settings))
        self.assertNotIn(secret, settings.fingerprint)
        self.assertEqual(len(settings.fingerprint), 64)

    def test_endpoint_and_model_are_validated_separately(self):
        settings = LLMSettings(base_url="https://gateway.example/v1")
        settings.validate_endpoint()
        with self.assertRaises(LLMConfigurationError):
            settings.validate_for_request()
        with self.assertRaises(LLMConfigurationError):
            LLMSettings(base_url="not-a-url", model="x").validate_endpoint()

    def test_default_retry_limits_are_three(self):
        self.assertEqual(LLMSettings().max_retries, 3)
        self.assertEqual(STRUCTURED_RESPONSE_MAX_RETRIES, 3)
        self.assertEqual(DEFAULT_STRUCTURED_OUTPUT_TOKENS, 2000)
        self.assertEqual(RETRY_OUTPUT_TOKEN_INCREMENT, 1000)
        self.assertEqual(VOCAB_IMPORT_OUTPUT_TOKENS, 20000)
        self.assertEqual(VOCAB_IMPORT_RETRY_TOKEN_INCREMENT, 2000)


class StructuredTaskTests(unittest.TestCase):
    def setUp(self):
        self.settings = LLMSettings(
            base_url="https://gateway.example/v1",
            model="model-a",
            api_key="sk-test-secret",
            reasoning_effort="disabled",
        )

    def test_generate_sentence_uses_card_versioned_prompt_and_json_mode(self):
        client = mock_client(
            json.dumps(
                {
                    "english_sentence": "I will abandon the broken plan.",
                    "chinese_translation": "我会放弃这个有问题的计划。",
                }
            )
        )
        result = LLMService(self.settings, client=client).generate_sentence(
            Card("abandon", "v", "放弃")
        )
        self.assertEqual(result.prompt_version, SENTENCE_PROMPT_VERSION)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertNotIn("temperature", kwargs)
        self.assertIn(SENTENCE_PROMPT_VERSION, kwargs["messages"][0]["content"])

    def test_invalid_first_json_is_repaired_exactly_once(self):
        client = mock_client(
            '{"english_sentence":"This omits it","chinese_translation":"缺词"}',
            '{"english_sentence":"We abandon it now.","chinese_translation":"我们现在放弃它。"}',
        )
        result = LLMService(self.settings, client=client).generate_sentence(
            Card("abandon", "v", "放弃")
        )
        self.assertIn("abandon", result.english_sentence)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_target_word_validation_uses_letter_boundaries(self):
        client = mock_client(
            '{"english_sentence":"The party starts now.","chinese_translation":"聚会现在开始。"}',
            '{"english_sentence":"Art can challenge us.","chinese_translation":"艺术可以挑战我们。"}',
        )
        result = LLMService(self.settings, client=client).generate_sentence(
            Card("art", "n", "艺术")
        )
        self.assertEqual(result.english_sentence, "Art can challenge us.")

    def test_four_invalid_responses_exhaust_three_retries(self):
        client = mock_client("not json", "still bad", "[]", "null")
        with self.assertRaises(LLMResponseValidationError) as captured:
            LLMService(self.settings, client=client).generate_sentence(
                Card("abandon", "v", "放弃")
            )
        self.assertEqual(client.chat.completions.create.call_count, 4)
        self.assertEqual(captured.exception.attempts, 4)
        self.assertIn("4 次尝试", str(captured.exception))

    def test_empty_length_response_retries_original_task_with_more_tokens(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            completion(
                None,
                finish_reason="length",
                completion_tokens=600,
                reasoning_tokens=600,
                request_id="request-first",
            ),
            completion(
                '{"english_sentence":"We abandon it now.",'
                '"chinese_translation":"我们现在放弃它。"}',
                finish_reason="stop",
                completion_tokens=31,
                reasoning_tokens=0,
                request_id="request-second",
            ),
        ]
        result = LLMService(self.settings, client=client).generate_sentence(
            Card("abandon", "v", "放弃")
        )
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(result.english_sentence, "We abandon it now.")
        self.assertEqual([call.kwargs["max_tokens"] for call in calls], [2000, 3000])
        self.assertIn(SENTENCE_PROMPT_VERSION, calls[1].kwargs["messages"][0]["content"])

    def test_final_empty_response_exposes_safe_diagnostics(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            completion(
                None,
                finish_reason="length",
                completion_tokens=2000 + 1000 * index,
                reasoning_tokens=2000 + 1000 * index,
                request_id=f"request-{index}",
            )
            for index in range(4)
        ]
        with self.assertRaises(LLMResponseValidationError) as captured:
            LLMService(self.settings, client=client).generate_sentence(
                Card("abandon", "v", "放弃")
            )
        error = captured.exception
        self.assertEqual(error.finish_reason, "length")
        self.assertEqual(error.completion_tokens, 5000)
        self.assertEqual(error.reasoning_tokens, 5000)
        self.assertEqual(error.attempts, 4)
        self.assertEqual(error.request_id, "request-3")

    def test_evaluation_uses_source_as_authority(self):
        client = mock_client('{"score":88,"feedback":"准确自然。"}')
        result = LLMService(self.settings, client=client).evaluate_translation(
            "The proposal is sound.", "这个提议是合理的。", "这个方案很可靠。"
        )
        self.assertEqual(result.score, 88)
        self.assertEqual(result.prompt_version, EVALUATION_PROMPT_VERSION)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("参考译文只是一个可接受版本", prompt)

    def test_vocab_import_returns_typed_preview_data(self):
        client = mock_client(
            json.dumps(
                {
                    "items": [{"word": "strain", "pos": "n", "meaning": "压力"}],
                    "rejected": [{"source": "???", "reason": "无法识别"}],
                },
                ensure_ascii=False,
            )
        )
        result = LLMService(self.settings, client=client).normalize_vocab_batch(
            "strain\n???"
        )
        self.assertEqual(result.items[0].word, "strain")
        self.assertEqual(result.prompt_version, VOCAB_IMPORT_PROMPT_VERSION)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 20000)

    def test_vocab_import_retries_increase_budget_by_two_thousand(self):
        valid = json.dumps(
            {
                "items": [{"word": "strain", "pos": "n", "meaning": "压力"}],
                "rejected": [],
            },
            ensure_ascii=False,
        )
        client = mock_client(None, valid)
        LLMService(self.settings, client=client).normalize_vocab_batch("strain")
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(
            [call.kwargs["max_tokens"] for call in calls],
            [20000, 22000],
        )

    def test_auto_effort_omits_reasoning_parameter(self):
        settings = LLMSettings(
            base_url="http://localhost:11434/v1",
            model="local-model",
            reasoning_effort="auto",
        )
        client = mock_client('{"score":75,"feedback":"基本准确。"}')
        LLMService(settings, client=client).evaluate_translation("a", "甲", "甲")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


class EndpointUtilityTests(unittest.TestCase):
    def setUp(self):
        self.settings = LLMSettings(
            base_url="https://gateway.example/v1",
            model="model-a",
            api_key="sk-private-1234",
        )

    def test_list_models_sorts_and_deduplicates(self):
        client = Mock()
        client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="z-model"), {"id": "a-model"}, {"id": "a-model"}]
        )
        self.assertEqual(
            LLMService(self.settings, client=client).list_models(),
            ["a-model", "z-model"],
        )

    def test_list_models_does_not_require_a_selected_model(self):
        client = Mock()
        client.models.list.return_value = SimpleNamespace(data=[{"id": "found"}])
        settings = LLMSettings(base_url=self.settings.base_url, api_key="key")
        self.assertEqual(LLMService(settings, client=client).list_models(), ["found"])

    def test_list_models_failure_is_redacted(self):
        client = Mock()
        client.models.list.side_effect = RuntimeError("bad sk-private-1234")
        with self.assertRaises(LLMServiceError) as captured:
            LLMService(self.settings, client=client).list_models()
        self.assertNotIn("sk-private-1234", str(captured.exception))

    def test_connection_result_reports_endpoint_without_secret(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError(
            "authentication failed for sk-private-1234"
        )
        result = LLMService(self.settings, client=client).test_connection()
        self.assertFalse(result.ok)
        self.assertNotIn("sk-private-1234", result.message)
        self.assertEqual(result.base_url, "https://gateway.example/v1")


if __name__ == "__main__":
    unittest.main()
