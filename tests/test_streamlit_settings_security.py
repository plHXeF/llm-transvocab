import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import llm_service
from app_settings import AppSettings, SettingsStore
from llm_service import LLMSettings
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitSettingsSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        vocabulary = self.root / "vocabularies.csv"
        shutil.copy2(PROJECT_ROOT / "vocabularies.csv", vocabulary)
        self.endpoint = "https://gateway.example/v1"
        self.environment_key = "sk-environment-security-test"
        self.local_key = "sk-local-security-test"
        self.patchers = [
            patch.object(config, "VOCAB_FILE", str(vocabulary)),
            patch.object(config, "APP_SETTINGS_FILE", self.root / "app_settings.json"),
            patch.object(config, "API_KEYS_FILE", self.root / "api_keys.json"),
            patch.object(config, "LEARNING_DB_FILE", self.root / "learning.db"),
            patch.object(
                config,
                "MODEL_ERROR_LOG_FILE",
                self.root / "model_errors.jsonl",
            ),
            patch.object(config, "DEFAULT_BASE_URL", self.endpoint),
            patch.object(config, "DEFAULT_MODEL", "seed-model"),
            patch.object(config, "DEFAULT_REASONING_EFFORT", "auto"),
            patch.object(config, "API_KEY", self.environment_key),
            patch.object(config, "DEEPSEEK_API_KEY", ""),
            patch.object(config, "OPENAI_API_KEY", ""),
        ]
        for patcher in self.patchers:
            patcher.start()
        self.store = SettingsStore()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def _run_app(self) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "vocab_web.py")).run(timeout=30)
        self.assertFalse(app.exception)
        return app

    def _save_local_key(self) -> None:
        self.store.save(
            AppSettings(
                llm=LLMSettings(
                    base_url=self.endpoint,
                    model="seed-model",
                    api_key=self.local_key,
                )
            )
        )

    def _assert_no_secret_in_text_inputs(self, app: AppTest) -> None:
        rendered_values = {str(widget.value) for widget in app.text_input}
        self.assertNotIn(self.local_key, rendered_values)
        self.assertNotIn(self.environment_key, rendered_values)

    def test_saved_and_environment_keys_never_hydrate_browser_widget(self):
        self._save_local_key()
        app = self._run_app()
        self.assertEqual(app.session_state["llm_settings"].api_key, self.local_key)
        self.assertEqual(app.session_state["settings_new_api_key"], "")
        self._assert_no_secret_in_text_inputs(app)

        app.radio(key="settings_key_action").set_value("replace").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="settings_new_api_key").value, "")
        self._assert_no_secret_in_text_inputs(app)

    def test_keep_can_change_model_without_copying_environment_key_to_disk(self):
        app = self._run_app()
        app.text_input(key="settings_manual_model").set_value("model-b")
        next(button for button in app.button if button.label == "应用设置").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        effective = app.session_state["llm_settings"]
        self.assertEqual(effective.model, "model-b")
        self.assertEqual(effective.api_key, self.environment_key)
        self.assertEqual(self.store.api_key_source(self.endpoint), "environment")
        self.assertFalse((self.root / "api_keys.json").exists())

    def test_delete_falls_back_to_environment_key_in_current_session(self):
        self._save_local_key()
        app = self._run_app()
        app.radio(key="settings_key_action").set_value("delete").run(timeout=30)
        next(button for button in app.button if button.label == "应用设置").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["llm_settings"].api_key, self.environment_key)
        self.assertEqual(self.store.api_key_source(self.endpoint), "environment")
        self.assertEqual(app.radio(key="settings_key_action").value, "keep")
        payload = json.loads(
            (self.root / "api_keys.json").read_text(encoding="utf-8")
        )
        self.assertFalse(payload["keys"])

    def test_provider_control_is_gone_and_base_url_is_editable(self):
        app = self._run_app()
        self.assertFalse(any(item.label == "模型提供商" for item in app.selectbox))
        base = app.text_input(key="settings_base_url")
        self.assertFalse(base.disabled)
        self.assertEqual(base.value, self.endpoint)

    def test_models_are_loaded_live_then_selected_from_searchable_list(self):
        class FakeLLMService:
            def __init__(self, settings):
                self.settings = settings

            def list_models(self):
                return ["alpha-model", "beta-model"]

        with patch.object(llm_service, "LLMService", FakeLLMService):
            app = self._run_app()
            next(
                button for button in app.button if button.label == "搜索可用模型"
            ).click()
            app.run(timeout=30)
            self.assertFalse(app.exception)
            model_select = app.selectbox(key="settings_selected_model")
            self.assertEqual(model_select.options, ["alpha-model", "beta-model"])
            model_select.set_value("beta-model").run(timeout=30)
            next(button for button in app.button if button.label == "应用设置").click()
            app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["llm_settings"].model, "beta-model")


if __name__ == "__main__":
    unittest.main()
