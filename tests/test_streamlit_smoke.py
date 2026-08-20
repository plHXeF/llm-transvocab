import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import llm_service
import pyarrow as pa
from domain import Card
from learning_store import LearningStore
from llm_service import (
    EvaluationResult,
    ImportBatchResult,
    SentenceResult,
    VocabularyItem,
)
from streamlit.testing.v1 import AppTest
from model_error_log import ModelErrorEntry
from vocab_web import _model_error_dataframe
from vocabulary_repository import VocabularyRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        vocabulary = root / "vocabularies.csv"
        shutil.copy2(PROJECT_ROOT / "vocabularies.csv", vocabulary)
        self.vocabulary_path = vocabulary
        self.patchers = [
            patch.object(config, "VOCAB_FILE", str(vocabulary)),
            patch.object(config, "APP_SETTINGS_FILE", root / "app_settings.json"),
            patch.object(config, "API_KEYS_FILE", root / "api_keys.json"),
            patch.object(config, "LEARNING_DB_FILE", root / "learning.db"),
            patch.object(config, "MODEL_ERROR_LOG_FILE", root / "model_errors.jsonl"),
            patch.object(config, "DEFAULT_BASE_URL", "https://gateway.example/v1"),
            patch.object(config, "DEFAULT_MODEL", "test-model"),
            patch.object(config, "DEFAULT_REASONING_EFFORT", "auto"),
            patch.object(config, "API_KEY", ""),
            patch.object(config, "DEEPSEEK_API_KEY", ""),
            patch.object(config, "OPENAI_API_KEY", ""),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def test_all_explicit_pages_render_without_model_calls(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("开始学习", [header.value for header in app.header])

        app.radio(key="page_navigation").set_value("学习数据").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("学习数据", [header.value for header in app.header])

        app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("词库管理", [header.value for header in app.header])

    def test_mixed_optional_model_diagnostics_are_arrow_safe(self):
        entries = [
            ModelErrorEntry(
                timestamp="2026-08-20T00:00:00Z",
                operation="sentence_generation",
                category="response_validation",
                error_type="ValueError",
                message="invalid",
                completion_tokens=None,
                attempts=None,
            ),
            ModelErrorEntry(
                timestamp="2026-08-20T00:00:01Z",
                operation="translation_evaluation",
                category="timeout",
                error_type="TimeoutError",
                message="timeout",
                status_code=504,
                completion_tokens=2000,
                reasoning_tokens=1000,
                latency_ms=4500,
                attempts=2,
            ),
        ]
        frame = _model_error_dataframe(entries)
        table = pa.Table.from_pandas(frame)
        self.assertEqual(table.num_rows, 2)
        self.assertTrue(all(str(dtype) == "string" for dtype in frame.dtypes))
        self.assertIn("—", frame["输出 token"].tolist())

    def test_sentence_difficulty_is_saved_and_used_for_generation(self):
        class FakeLLMService:
            difficulties = []

            def __init__(self, settings):
                self.settings = settings

            def generate_sentence(
                self, card, register="general", difficulty="cet6_postgrad"
            ):
                self.difficulties.append(str(getattr(difficulty, "value", difficulty)))
                return SentenceResult(
                    english_sentence=f"I will use {card.word} correctly today.",
                    chinese_translation=f"我今天会正确使用 {card.word}。",
                )

        with patch.object(llm_service, "LLMService", FakeLLMService):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
            app.selectbox(key="sentence_difficulty").set_value("ielts").run(
                timeout=30
            )
            self.assertFalse(app.exception)
            payload = json.loads(
                Path(config.APP_SETTINGS_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["sentence_difficulty"], "ielts")
            self.assertEqual(app.session_state["config_revision"], 1)

            next(button for button in app.button if button.label == "开始本轮").click()
            app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(FakeLLMService.difficulties)
        self.assertTrue(
            all(value == "ielts" for value in FakeLLMService.difficulties)
        )

    def test_learning_flow_records_score_without_a_real_api(self):
        class FakeLLMService:
            def __init__(self, settings):
                self.settings = settings

            def generate_sentence(
                self, card, register="general", difficulty="cet6_postgrad"
            ):
                return SentenceResult(
                    english_sentence=f"I will use {card.word} correctly today.",
                    chinese_translation=f"我今天会正确使用 {card.word}。",
                )

            def evaluate_translation(self, original, reference, answer, *, card=None):
                return EvaluationResult(
                    score=88,
                    feedback="意思准确，表达自然。",
                    target_error_weight=0.1,
                    attribution_confidence=0.9,
                )

        with patch.object(llm_service, "LLMService", FakeLLMService):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
            next(button for button in app.button if button.label == "开始本轮").click()
            app.run(timeout=30)
            self.assertFalse(app.exception)

            app.text_area[0].set_value("我会正确翻译这个句子。")
            next(button for button in app.button if button.label == "提交翻译").click()
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any("88" in markdown.value for markdown in app.markdown))

    def test_meaning_can_be_revealed_once_and_forces_target_attribution(self):
        class FakeLLMService:
            def __init__(self, settings):
                self.settings = settings

            def generate_sentence(
                self, card, register="general", difficulty="cet6_postgrad"
            ):
                return SentenceResult(
                    english_sentence=f"I will use {card.word} correctly today.",
                    chinese_translation=f"我今天会正确使用 {card.word}。",
                )

            def evaluate_translation(self, original, reference, answer, *, card=None):
                return EvaluationResult(
                    score=70,
                    feedback="上下文部分有误。",
                    target_error_weight=0.0,
                    attribution_confidence=0.9,
                    non_target_error_tags=("context_vocabulary",),
                )

        with patch.object(llm_service, "LLMService", FakeLLMService):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
            next(button for button in app.button if button.label == "开始本轮").click()
            app.run(timeout=30)

            next(button for button in app.button if button.label == "查看释义").click()
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(
                any("剩余 5 秒" in caption.value for caption in app.caption)
            )

            next(
                button for button in app.button if button.label == "收起释义"
            ).click()
            app.run(timeout=30)
            locked = next(
                button
                for button in app.button
                if button.label == "释义已查看"
            )
            self.assertTrue(locked.disabled)
            self.assertFalse(
                any(button.label == "查看释义" for button in app.button)
            )

            app.text_area[0].set_value("我会翻译这个句子。")
            next(button for button in app.button if button.label == "提交翻译").click()
            app.run(timeout=30)
            self.assertFalse(app.exception)
            event = LearningStore(config.LEARNING_DB_FILE).list_review_events()[-1]
            self.assertEqual(event.target_error_weight, 1.0)
            self.assertEqual(event.attribution_confidence, 1.0)
            self.assertTrue(event.meaning_revealed)
            self.assertEqual(event.feedback, "上下文部分有误。")

    def test_meaning_auto_closes_after_deadline_without_waiting(self):
        class FakeLLMService:
            def __init__(self, settings):
                self.settings = settings

            def generate_sentence(
                self, card, register="general", difficulty="cet6_postgrad"
            ):
                return SentenceResult(
                    english_sentence=f"I will use {card.word} correctly today.",
                    chinese_translation=f"我今天会正确使用 {card.word}。",
                )

        with patch.object(llm_service, "LLMService", FakeLLMService):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(
                timeout=30
            )
            next(button for button in app.button if button.label == "开始本轮").click()
            app.run(timeout=30)
            next(button for button in app.button if button.label == "查看释义").click()
            app.run(timeout=30)
            app.session_state["meaning_reveal_deadline"] = 0.0
            app.run(timeout=30)

        self.assertFalse(app.exception)
        locked = next(
            button for button in app.button if button.label == "释义已查看"
        )
        self.assertTrue(locked.disabled)
        self.assertTrue(app.session_state["meaning_revealed"])
        self.assertFalse(app.session_state["meaning_visible"])

    def test_invalid_local_import_shows_diagnostics_without_preview_rows(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)

        next(
            area for area in app.text_area if area.label == "或粘贴文本"
        ).set_value("only-one-field")
        next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.label.startswith("非标准文本使用当前模型")
        ).uncheck()
        next(
            button for button in app.button if button.label == "解析并生成预览"
        ).click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("应有 3 个字段" in warning.value for warning in app.warning)
        )
        self.assertTrue(
            any("没有可导入的有效词条" in info.value for info in app.info)
        )

    def test_structured_pasted_csv_uses_local_parser_even_when_ai_is_enabled(self):
        class ModelMustNotBeCalled:
            def __init__(self, settings):
                raise AssertionError("structured CSV must not call the model")

        with patch.object(llm_service, "LLMService", ModelMustNotBeCalled):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(
                timeout=30
            )
            app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)
            next(
                area for area in app.text_area if area.label == "或粘贴文本"
            ).set_value("word,pos,meaning\nzz-local-csv,n,本地解析")
            next(
                button for button in app.button if button.label == "解析并生成预览"
            ).click()
            app.run(timeout=30)

        self.assertFalse(app.exception)
        metrics = {metric.label: str(metric.value) for metric in app.metric}
        self.assertEqual("1", metrics["将新增"])

    def test_suspicious_whitespace_triples_are_cleaned_by_model(self):
        class FakeImportService:
            calls = []

            def __init__(self, settings):
                pass

            def normalize_vocab_batch(self, raw_content):
                self.calls.append(raw_content)
                return ImportBatchResult(
                    items=(
                        VocabularyItem("zz-ai-cleaned-one", "adj", "测试释义一"),
                        VocabularyItem("zz-ai-cleaned-two", "adj", "测试释义二"),
                    ),
                    rejected=(),
                )

        with patch.object(llm_service, "LLMService", FakeImportService):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(
                timeout=30
            )
            app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)
            next(
                area for area in app.text_area if area.label == "或粘贴文本"
            ).set_value(
                "pellucid —— adjective，清澈透明\n"
                "recondite: very difficult to understand"
            )
            next(
                button for button in app.button if button.label == "解析并生成预览"
            ).click()
            app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(len(FakeImportService.calls), 1)
        metrics = {metric.label: str(metric.value) for metric in app.metric}
        self.assertEqual("2", metrics["将新增"])

    def test_import_preview_cancel_and_confirm_do_not_crash(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)

        def fill_manual_card(target: AppTest, word: str, meaning: str) -> None:
            next(item for item in target.text_input if item.label == "单词").set_value(word)
            next(item for item in target.text_input if item.label == "词性").set_value("n")
            next(item for item in target.text_input if item.label == "中文释义").set_value(
                meaning
            )
            next(
                button for button in target.button if button.label == "准备预览"
            ).click()
            target.run(timeout=30)

        cancelled_word = "zz-app-test-cancelled"
        fill_manual_card(app, cancelled_word, "取消测试")
        next(button for button in app.button if button.label == "取消预览").click()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertNotIn(
            cancelled_word,
            {card.word for card in VocabularyRepository(self.vocabulary_path).load().cards},
        )

        # A fresh AppTest instance avoids retaining removed dynamic-widget nodes
        # from the cancelled preview in Streamlit's testing element tree.
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)
        committed_word = "zz-app-test-committed"
        fill_manual_card(app, committed_word, "确认测试")
        next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.label == "我已检查预览内容"
        ).check()
        app.run(timeout=30)
        next(
            button for button in app.button if button.label == "确认批量导入"
        ).click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIn(
            committed_word,
            {card.word for card in VocabularyRepository(self.vocabulary_path).load().cards},
        )

    def test_nan_and_empty_editor_rows_are_not_import_candidates(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.radio(key="page_navigation").set_value("词库管理").run(timeout=30)
        app.session_state["import_rows"] = [
            {"word": float("nan"), "pos": "n", "meaning": "不应导入"},
            {"word": "", "pos": "", "meaning": ""},
        ]
        app.session_state["import_preview_revision"] += 1
        app.run(timeout=30)

        self.assertFalse(app.exception)
        metrics = {metric.label: str(metric.value) for metric in app.metric}
        self.assertEqual("0", metrics["有效行"])
        self.assertEqual("0", metrics["将新增"])
        self.assertNotIn(
            "nan",
            {card.word for card in VocabularyRepository(self.vocabulary_path).load().cards},
        )

    def test_confirmed_progress_reset_does_not_raise_streamlit_api_exception(self):
        card = Card("appeal", "n", "呼吁")
        store = LearningStore(config.LEARNING_DB_FILE)
        store.record_review(card, 90, review_key="seed:reset")

        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.checkbox(key="confirm_data_reset").set_value(True).run(timeout=30)
        next(
            button for button in app.button if button.label == "重置熟练度/遗忘权重"
        ).click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIsNone(store.get_progress(card.card_id))
        self.assertEqual(len(store.list_review_events()), 1)
        self.assertTrue(any("历史仍保留" in message.value for message in app.success))

    def test_confirmed_history_clear_does_not_raise_streamlit_api_exception(self):
        card = Card("appeal", "n", "呼吁")
        store = LearningStore(config.LEARNING_DB_FILE)
        expected_progress = store.record_review(card, 90, review_key="seed:history")

        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.checkbox(key="confirm_data_reset").set_value(True).run(timeout=30)
        next(button for button in app.button if button.label == "清空学习历史").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(store.get_progress(card.card_id), expected_progress)
        self.assertEqual(store.list_review_events(), [])
        self.assertTrue(any("当前熟练度仍保留" in message.value for message in app.success))

    @staticmethod
    def _prime_active_snapshot(app: AppTest, *, batch_id: str) -> None:
        snapshot = Card("snapshot", "n", "批次快照")
        app.session_state["learning_active"] = True
        app.session_state["batch_complete"] = False
        app.session_state["batch_cards"] = [snapshot]
        app.session_state["batch_id"] = batch_id
        app.session_state["batch_vocabulary_fingerprint"] = "original-fingerprint"
        app.session_state["current_index"] = 0
        app.session_state["current_word_data"] = SentenceResult(
            english_sentence="This sentence comes from the active batch snapshot.",
            chinese_translation="这个句子来自活动批次快照。",
        )

    def _assert_snapshot_is_rendered(self, app: AppTest) -> None:
        self.assertFalse(app.exception)
        self.assertIn("snapshot", [item.value for item in app.subheader])
        self.assertTrue(
            any("本轮继续使用启动时快照" in item.value for item in app.warning)
        )
        self.assertFalse(
            any("当前没有可学习的有效词条" in item.value for item in app.warning)
        )

    def test_active_batch_survives_an_emptied_vocabulary_file(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        self._prime_active_snapshot(app, batch_id="snapshot-batch-empty")
        self.vocabulary_path.write_text("word,pos,meaning\n", encoding="utf-8")
        app.run(timeout=30)

        self._assert_snapshot_is_rendered(app)

    def test_active_batch_survives_a_deleted_vocabulary_file(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        self._prime_active_snapshot(app, batch_id="snapshot-batch-deleted")
        self.vocabulary_path.unlink()
        app.run(timeout=30)

        self._assert_snapshot_is_rendered(app)

    def test_skip_only_statistics_show_no_numeric_historical_average(self):
        card = Card("appeal", "n", "呼吁")
        LearningStore(config.LEARNING_DB_FILE).record_review(
            card,
            None,
            status="skipped",
            review_key="seed:skip-only",
        )

        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        app.radio(key="page_navigation").set_value("学习数据").run(timeout=30)

        self.assertFalse(app.exception)
        metrics = {metric.label: str(metric.value) for metric in app.metric}
        self.assertEqual(metrics["历史平均分"], "—")
        self.assertNotEqual(metrics["历史平均分"], "0.0")
        self.assertTrue(
            any("目前只有跳过记录" in message.value for message in app.info)
        )

    def test_model_failure_is_logged_redacted_and_can_be_cleared(self):
        secret = "sk-streamlit-log-secret-1234"

        class FailingLLMService:
            def __init__(self, settings):
                self.settings = settings

            def generate_sentence(
                self, card, register="general", difficulty="cet6_postgrad"
            ):
                raise llm_service.LLMServiceError(
                    f"upstream rejected {secret}",
                    category="api_status",
                    status_code=503,
                    request_id="request-test-1",
                )

        with patch.object(config, "API_KEY", secret), patch.object(
            llm_service, "LLMService", FailingLLMService
        ):
            app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
            next(button for button in app.button if button.label == "开始本轮").click()
            app.run(timeout=30)

        self.assertFalse(app.exception)
        log_path = Path(config.MODEL_ERROR_LOG_FILE)
        payload = log_path.read_text(encoding="utf-8")
        self.assertNotIn(secret, payload)
        self.assertIn("sentence_generation", payload)
        self.assertIn("request-test-1", payload)
        self.assertTrue(any("例句生成失败" in item.value for item in app.error))

        app.checkbox(key="confirm_clear_model_errors").set_value(True).run(timeout=30)
        next(button for button in app.button if button.label == "清空错误日志").click()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(log_path.read_text(encoding="utf-8"), "")
        self.assertTrue(any("已清空 1 条" in item.value for item in app.success))


if __name__ == "__main__":
    unittest.main()
