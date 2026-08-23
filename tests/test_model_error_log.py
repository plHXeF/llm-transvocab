import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_service import LLMResponseValidationError, LLMSettings
from model_error_log import ModelErrorLog


class ModelErrorLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "nested" / "model_errors.jsonl"
        self.secret = "sk-super-secret-value-1234"
        self.settings = LLMSettings(
            base_url=(
                "https://user:password@gateway.example:8443/private/v1"
                "?api_key=query-secret"
            ),
            model="model-a",
            api_key=self.secret,
            reasoning_effort="high",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_record_is_owner_only_and_redacts_secrets_and_endpoint(self):
        error = RuntimeError(
            f"Authorization: Bearer bearer-secret; api_key=other-secret; {self.secret}"
        )
        entry = ModelErrorLog(self.path).record(
            "sentence_generation",
            error,
            self.settings,
            prompt_version="sentence.v1",
            card_id="card:abc",
            batch_id="batch-1",
        )

        payload = self.path.read_text(encoding="utf-8")
        self.assertNotIn(self.secret, payload)
        self.assertNotIn("bearer-secret", payload)
        self.assertNotIn("other-secret", payload)
        self.assertNotIn("password", payload)
        self.assertNotIn("query-secret", payload)
        self.assertEqual(entry.endpoint, "https://gateway.example:8443")
        if os.name != "nt":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_model_and_request_id_are_redacted_too(self):
        settings = LLMSettings(
            base_url="https://gateway.example/v1",
            model=self.secret,
            api_key=self.secret,
        )
        entry = ModelErrorLog(self.path).record(
            "connection_test",
            RuntimeError("failed"),
            settings,
            request_id=f"trace-{self.secret}",
        )
        self.assertNotIn(self.secret, entry.model)
        self.assertNotIn(self.secret, entry.request_id or "")
        self.assertNotIn(self.secret, self.path.read_text(encoding="utf-8"))

    def test_structured_error_metadata_and_validation_category_are_kept(self):
        error = LLMResponseValidationError(
            "invalid result",
            finish_reason="length",
            completion_tokens=500,
            reasoning_tokens=500,
            latency_ms=4800,
            attempts=4,
            request_id="request-safe",
        )
        entry = ModelErrorLog(self.path).record(
            "translation_evaluation", error, self.settings
        )
        self.assertEqual(entry.category, "response_validation")
        self.assertEqual(entry.error_type, "LLMResponseValidationError")
        self.assertEqual(entry.reasoning_effort, "high")
        self.assertEqual(entry.finish_reason, "length")
        self.assertEqual(entry.completion_tokens, 500)
        self.assertEqual(entry.reasoning_tokens, 500)
        self.assertEqual(entry.latency_ms, 4800)
        self.assertEqual(entry.attempts, 4)
        self.assertEqual(entry.request_id, "request-safe")

    def test_old_log_entry_without_new_diagnostics_is_still_readable(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-08-16T00:00:00Z",
                    "operation": "sentence_generation",
                    "category": "response_validation",
                    "error_type": "ValueError",
                    "message": "invalid",
                    "schema_version": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        entry = ModelErrorLog(self.path).list_entries()[0]
        self.assertIsNone(entry.finish_reason)
        self.assertIsNone(entry.completion_tokens)
        self.assertEqual(entry.schema_version, 2)

    def test_bounded_tolerant_read_export_and_clear(self):
        log = ModelErrorLog(self.path, max_entries=2)
        for index in range(3):
            log.record(f"operation-{index}", RuntimeError(f"error-{index}"))
        entries = log.list_entries()
        self.assertEqual([item.operation for item in entries], ["operation-1", "operation-2"])

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("not-json\n")
        self.assertEqual(len(log.list_entries()), 2)
        exported = [json.loads(line) for line in log.export_jsonl().splitlines()]
        self.assertEqual(len(exported), 2)

        self.assertEqual(log.clear(), 2)
        self.assertEqual(log.list_entries(), [])
        self.assertEqual(self.path.read_text(encoding="utf-8"), "")
        if os.name != "nt":
            self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
