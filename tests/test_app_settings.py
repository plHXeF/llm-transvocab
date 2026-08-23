import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app_settings import AppSettings, SettingsError, SettingsStore
from llm_service import LLMSettings, ReasoningEffort, SentenceDifficulty


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.path = root / "app_settings.json"
        self.keys_path = root / "provider_keys.json"
        self.store = SettingsStore(self.path, self.keys_path)
        self.endpoint = "https://gateway.example/v1"
        self.secret = "sk-this-is-a-private-key"
        self.settings = AppSettings(
            llm=LLMSettings(
                base_url=self.endpoint,
                model="model-a",
                api_key=self.secret,
                reasoning_effort="medium",
            ),
            sentence_difficulty=SentenceDifficulty.IELTS,
        )
        self.patchers = [
            patch("app_settings.config.API_KEY", ""),
            patch("app_settings.config.DEEPSEEK_API_KEY", ""),
            patch("app_settings.config.OPENAI_API_KEY", ""),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def test_round_trip_is_atomic_owner_only_and_split(self):
        self.store.save(self.settings)
        loaded = self.store.load()

        self.assertEqual(loaded.llm.base_url, self.endpoint)
        self.assertEqual(loaded.llm.model, "model-a")
        self.assertEqual(loaded.llm.reasoning_effort, ReasoningEffort.MEDIUM)
        self.assertEqual(loaded.llm.api_key, self.secret)
        self.assertEqual(loaded.sentence_difficulty, SentenceDifficulty.IELTS)
        if os.name != "nt":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.keys_path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(self.secret, self.path.read_text(encoding="utf-8"))
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["sentence_difficulty"], "ielts")
        self.assertNotIn("provider", payload["llm"])

    def test_missing_settings_returns_none(self):
        self.assertIsNone(self.store.load())

    def test_safe_views_and_repr_never_expose_key(self):
        self.assertNotIn(self.secret, repr(self.settings))
        self.assertNotIn(self.secret, repr(self.settings.llm))
        self.assertNotIn(self.secret, json.dumps(self.settings.safe_dict()))

    def test_endpoint_keys_are_isolated_by_full_base_url(self):
        first = self.settings
        second = AppSettings(
            llm=LLMSettings(
                base_url="https://gateway.example/other/v1",
                model="model-b",
                api_key="key-for-b",
            )
        )
        self.store.save(first)
        self.store.save(second)

        self.assertEqual(self.store.load_api_key(first.llm.base_url), self.secret)
        self.assertEqual(self.store.load_api_key(second.llm.base_url), "key-for-b")
        self.assertEqual(self.store.api_key_source(first.llm.base_url), "local")

    def test_environment_key_is_fallback_for_any_endpoint(self):
        with patch("app_settings.config.API_KEY", "sk-from-environment"):
            self.assertEqual(
                self.store.load_api_key("https://new.example/v1"),
                "sk-from-environment",
            )
            self.assertEqual(
                self.store.api_key_source("https://new.example/v1"),
                "environment",
            )

    def test_legacy_environment_keys_follow_matching_urls(self):
        import config

        with patch("app_settings.config.DEEPSEEK_API_KEY", "deepseek-env"):
            self.assertEqual(
                self.store.load_api_key(config.DEEPSEEK_BASE_URL), "deepseek-env"
            )
        with patch("app_settings.config.OPENAI_API_KEY", "openai-env"):
            self.assertEqual(
                self.store.load_api_key(config.OPENAI_BASE_URL), "openai-env"
            )

    def test_keep_changes_settings_without_touching_key_file(self):
        changed = AppSettings(
            llm=LLMSettings(
                base_url=self.endpoint,
                model="model-b",
                api_key="must-not-be-persisted",
            )
        )
        self.store.save(changed, key_action="keep")
        self.assertFalse(self.keys_path.exists())
        self.assertEqual(self.store.load().llm.model, "model-b")
        self.assertEqual(self.store.load().llm.api_key, "")

    def test_replace_then_delete_reveals_environment_fallback(self):
        self.store.save(self.settings, key_action="replace")
        with patch("app_settings.config.API_KEY", "environment-key"):
            self.store.save(self.settings, key_action="delete")
            self.assertEqual(self.store.api_key_source(self.endpoint), "environment")
            self.assertEqual(self.store.load_api_key(self.endpoint), "environment-key")
        self.assertNotIn(self.secret, self.keys_path.read_text(encoding="utf-8"))

    def test_delete_isolated_to_selected_endpoint(self):
        other = AppSettings(
            llm=LLMSettings(
                base_url="http://localhost:8002/v1",
                model="local-b",
                api_key="key-for-b",
            )
        )
        self.store.save(self.settings)
        self.store.save(other)
        self.store.save(self.settings, key_action="delete")
        self.assertEqual(self.store.api_key_source(self.endpoint), "none")
        self.assertEqual(self.store.load_api_key(other.llm.base_url), "key-for-b")

    def test_invalid_key_action_is_rejected_without_writes(self):
        with self.assertRaises(SettingsError):
            self.store.save(self.settings, key_action="invalid")  # type: ignore[arg-type]
        self.assertFalse(self.path.exists())
        self.assertFalse(self.keys_path.exists())

    def test_v1_provider_settings_and_key_are_loaded_and_migrated(self):
        import config

        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "llm": {
                        "provider": "deepseek",
                        "model": "legacy-model",
                        "base_url": None,
                        "reasoning_effort": "disabled",
                    },
                }
            ),
            encoding="utf-8",
        )
        self.keys_path.write_text(
            json.dumps({"version": 1, "keys": {"deepseek": "legacy-key"}}),
            encoding="utf-8",
        )

        loaded = self.store.load()
        self.assertEqual(loaded.llm.base_url, config.DEEPSEEK_BASE_URL)
        self.assertEqual(loaded.llm.model, "legacy-model")
        self.assertEqual(loaded.llm.api_key, "legacy-key")
        self.store.save(loaded, key_action="keep")
        migrated = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(migrated["version"], 3)
        self.assertEqual(
            loaded.sentence_difficulty, SentenceDifficulty.CET6_POSTGRAD
        )
        self.assertNotIn("provider", migrated["llm"])

    def test_v2_settings_default_to_cet6_postgrad(self):
        self.path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "llm": {
                        "model": "previous-model",
                        "base_url": self.endpoint,
                        "reasoning_effort": "disabled",
                    },
                }
            ),
            encoding="utf-8",
        )
        loaded = self.store.load()
        self.assertEqual(
            loaded.sentence_difficulty, SentenceDifficulty.CET6_POSTGRAD
        )

    def test_legacy_custom_endpoint_slot_is_still_read(self):
        import hashlib

        digest = hashlib.sha256(self.endpoint.encode("utf-8")).hexdigest()[:24]
        self.keys_path.write_text(
            json.dumps(
                {"version": 1, "keys": {f"custom:{digest}": "legacy-custom"}}
            ),
            encoding="utf-8",
        )
        self.assertEqual(self.store.load_api_key(self.endpoint), "legacy-custom")

    def test_default_store_reads_legacy_key_file_when_new_file_is_absent(self):
        import config

        root = Path(self.temporary.name)
        new_keys = root / "api_keys.json"
        legacy_keys = root / "provider_keys.json"
        legacy_keys.write_text(
            json.dumps({"version": 1, "keys": {"deepseek": "legacy-key"}}),
            encoding="utf-8",
        )
        with (
            patch.object(config, "APP_SETTINGS_FILE", root / "settings.json"),
            patch.object(config, "API_KEYS_FILE", new_keys),
            patch.object(config, "LEGACY_PROVIDER_KEYS_FILE", legacy_keys),
        ):
            store = SettingsStore()
            self.assertEqual(
                store.load_api_key(config.DEEPSEEK_BASE_URL), "legacy-key"
            )
            store.save(
                AppSettings(
                    llm=LLMSettings(
                        base_url=config.DEEPSEEK_BASE_URL,
                        model="migrated-model",
                        api_key="replacement",
                    )
                )
            )
        self.assertTrue(new_keys.exists())
        self.assertIn("replacement", new_keys.read_text(encoding="utf-8"))

    @unittest.skipIf(os.name == "nt", "Windows does not expose POSIX mode bits")
    def test_load_tightens_unsafe_file_modes(self):
        self.store.save(self.settings)
        os.chmod(self.path, 0o644)
        os.chmod(self.keys_path, 0o644)
        self.store.load()
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.keys_path.stat().st_mode & 0o777, 0o600)

    def test_unknown_settings_version_is_rejected(self):
        self.path.write_text('{"version":99,"llm":{}}', encoding="utf-8")
        with self.assertRaises(SettingsError):
            self.store.load()


if __name__ == "__main__":
    unittest.main()
