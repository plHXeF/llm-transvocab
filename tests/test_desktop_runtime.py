import tempfile
import unittest
from pathlib import Path

from desktop_runtime import configure_desktop_environment, user_data_directory


class DesktopRuntimeTests(unittest.TestCase):
    def test_platform_user_data_directories(self):
        home = Path("/users/tester")
        self.assertEqual(
            user_data_directory(platform="darwin", environ={}, home=home),
            home / "Library" / "Application Support" / "LLM TransVocab",
        )
        self.assertEqual(
            user_data_directory(
                platform="win32",
                environ={"LOCALAPPDATA": "C:/Local"},
                home=home,
            ),
            Path("C:/Local") / "LLM TransVocab",
        )
        self.assertEqual(
            user_data_directory(
                platform="linux",
                environ={"XDG_DATA_HOME": "/var/user-data"},
                home=home,
            ),
            Path("/var/user-data") / "LLM TransVocab",
        )

    def test_first_launch_copies_vocabulary_and_sets_existing_env_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "vocabularies.csv").write_text(
                "word,pos,meaning\nhello,n,你好\n", encoding="utf-8"
            )
            environment = {"LOCALAPPDATA": str(root / "local")}

            paths = configure_desktop_environment(
                environ=environment,
                platform="win32",
                home=root / "home",
                bundled_directory=bundle,
            )

            self.assertEqual(
                paths.vocabulary.read_text(encoding="utf-8"),
                "word,pos,meaning\nhello,n,你好\n",
            )
            self.assertEqual(environment["VOCAB_FILE"], str(paths.vocabulary))
            self.assertEqual(
                environment["VOCAB_LEARNING_DB_FILE"], str(paths.learning_db)
            )
            self.assertEqual(environment["VOCAB_DESKTOP_MODE"], "1")

    def test_later_launch_does_not_overwrite_user_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            bundled_vocab = bundle / "vocabularies.csv"
            bundled_vocab.write_text("word,pos,meaning\nfirst,n,一\n", encoding="utf-8")
            environment = {"XDG_DATA_HOME": str(root / "data")}
            paths = configure_desktop_environment(
                environ=environment,
                platform="linux",
                home=root / "home",
                bundled_directory=bundle,
            )
            paths.vocabulary.write_text(
                "word,pos,meaning\npersonal,n,个人\n", encoding="utf-8"
            )
            bundled_vocab.write_text("word,pos,meaning\nsecond,n,二\n", encoding="utf-8")

            configure_desktop_environment(
                environ=environment,
                platform="linux",
                home=root / "home",
                bundled_directory=bundle,
            )

            self.assertIn("personal", paths.vocabulary.read_text(encoding="utf-8"))

    def test_explicit_vocabulary_path_is_respected_and_initialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "vocabularies.csv").write_text(
                "word,pos,meaning\ncustom,n,自定义\n", encoding="utf-8"
            )
            target = root / "elsewhere" / "words.csv"
            environment = {
                "XDG_DATA_HOME": str(root / "data"),
                "VOCAB_FILE": str(target),
            }

            paths = configure_desktop_environment(
                environ=environment,
                platform="linux",
                home=root / "home",
                bundled_directory=bundle,
            )

            self.assertEqual(paths.vocabulary, target)
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
