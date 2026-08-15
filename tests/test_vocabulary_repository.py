from __future__ import annotations

import codecs
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from domain import Card
from vocabulary_repository import (
    VocabularyRepository,
    parse_csv,
    parse_import_content,
    parse_text,
)


def issue_codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


class CsvParsingTests(unittest.TestCase):
    def test_loads_utf8_utf8_sig_and_gb18030(self) -> None:
        text = "word,pos,meaning\nabandon,v,放弃\n"
        variants = {
            "utf-8": text.encode("utf-8"),
            "utf-8-sig": codecs.BOM_UTF8 + text.encode("utf-8"),
            "gb18030": text.encode("gb18030"),
        }

        for expected_encoding, content in variants.items():
            with self.subTest(encoding=expected_encoding):
                result = parse_csv(content)
                self.assertEqual(expected_encoding, result.encoding)
                self.assertEqual(["abandon"], [card.word for card in result.cards])
                self.assertFalse(result.has_errors)

    def test_missing_columns_is_reported_without_raising(self) -> None:
        result = parse_csv("word,meaning\nabandon,放弃\n")

        self.assertEqual((), result.cards)
        self.assertIn("missing_columns", issue_codes(result))
        self.assertTrue(result.has_errors)

    def test_bad_rows_are_skipped_and_later_rows_still_load(self) -> None:
        result = parse_csv(
            "word,pos,meaning\n"
            "valid,n,有效的\n"
            "too-few,n\n"
            "empty,,空字段\n"
            "later,v,继续\n"
        )

        self.assertEqual(["valid", "later"], [card.word for card in result.cards])
        self.assertEqual(4, result.total_rows)
        self.assertIn("invalid_column_count", issue_codes(result))
        self.assertIn("empty_field", issue_codes(result))

    def test_malformed_csv_returns_partial_result_and_issue(self) -> None:
        result = parse_csv(
            'word,pos,meaning\nvalid,n,有效的\n"unterminated,n,坏行\n'
        )

        self.assertEqual(["valid"], [card.word for card in result.cards])
        self.assertIn("csv_error", issue_codes(result))

    def test_exact_normalised_triples_are_deduplicated_but_senses_remain(self) -> None:
        result = parse_csv(
            "word,pos,meaning\n"
            "strain,n,品种\n"
            "  STRAIN  ,N,品种\n"
            "strain,n,压力\n"
        )

        self.assertEqual(2, len(result.cards))
        self.assertEqual(["品种", "压力"], [card.meaning for card in result.cards])
        self.assertEqual(1, len(result.duplicates))
        self.assertIn("duplicate_in_source", issue_codes(result))
        self.assertEqual(Card("strain", "n", "品种").card_id, result.cards[0].card_id)

    def test_garbage_bytes_are_reported_when_no_supported_encoding_matches(self) -> None:
        # A lone 0x80 followed by an incomplete GB18030 sequence is invalid in
        # both UTF-8 and GB18030.
        result = parse_csv(b"\x80\x81")

        self.assertEqual((), result.cards)
        self.assertIn("decode_error", issue_codes(result))


class TextParsingTests(unittest.TestCase):
    def test_tab_text_with_header(self) -> None:
        result = parse_text("word\tpos\tmeaning\nabandon\tv\t放弃\n")

        self.assertEqual([Card("abandon", "v", "放弃")], list(result.cards))
        self.assertEqual(1, result.total_rows)

    def test_pipe_text_without_header(self) -> None:
        result = parse_text("strain|n|品种\nstrain|n|压力\n")

        self.assertEqual(2, len(result.cards))
        self.assertEqual(["品种", "压力"], [card.meaning for card in result.cards])

    def test_whitespace_text_keeps_spaces_in_meaning(self) -> None:
        result = parse_text("abandon v give up completely\n")

        self.assertEqual("give up completely", result.cards[0].meaning)

    def test_ascii_comma_in_whitespace_meaning_is_not_mistaken_for_delimiter(self) -> None:
        result = parse_text("abandon v give up, desert\n")

        self.assertEqual("give up, desert", result.cards[0].meaning)
        self.assertFalse(result.has_errors)

    def test_pasted_comma_rows_support_csv_quoting(self) -> None:
        result = parse_import_content(
            'word,pos,meaning\nabandon,v,"放弃,抛弃"\n',
            source_format="paste",
        )

        self.assertEqual("text", result.source_format)
        self.assertEqual("放弃,抛弃", result.cards[0].meaning)

    def test_format_is_inferred_from_filename(self) -> None:
        csv_result = parse_import_content(
            "meaning,word,pos\n放弃,abandon,v\n",
            source_name="words.CSV",
        )
        text_result = parse_import_content(
            "abandon\tv\t放弃\n",
            source_name="words.txt",
        )

        self.assertEqual("abandon", csv_result.cards[0].word)
        self.assertEqual("abandon", text_result.cards[0].word)


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "vocabularies.csv"
        self.repository = VocabularyRepository(self.path)

    def test_missing_file_is_an_empty_non_crashing_load(self) -> None:
        result = self.repository.load()

        self.assertEqual((), result.cards)
        self.assertIsNone(result.fingerprint)
        self.assertIn("file_not_found", issue_codes(result))

    def test_repository_load_reports_encoding_and_fingerprint(self) -> None:
        self.path.write_bytes("word,pos,meaning\nabandon,v,放弃\n".encode("gb18030"))

        result = self.repository.load()

        self.assertEqual("gb18030", result.encoding)
        self.assertEqual(self.repository.fingerprint(), result.fingerprint)
        self.assertEqual(64, len(result.fingerprint or ""))

    def test_preview_is_side_effect_free_and_separates_duplicate_kinds(self) -> None:
        original = Card("strain", "n", "品种")
        self.assertTrue(self.repository.write_cards([original]).success)
        before = self.repository.fingerprint()

        preview = self.repository.preview_import(
            "word,pos,meaning\n"
            "strain,n,品种\n"
            "strain,n,压力\n"
            "strain,n,压力\n",
            source_format="csv",
        )

        self.assertEqual(before, self.repository.fingerprint())
        self.assertEqual(["压力"], [card.meaning for card in preview.cards_to_add])
        self.assertEqual(1, len(preview.duplicates_in_source))
        self.assertEqual(1, len(preview.duplicates_in_library))
        self.assertTrue(preview.can_commit)

    def test_commit_preview_atomically_appends_and_preserves_order(self) -> None:
        original = Card("abandon", "v", "放弃")
        addition = Card("strain", "n", "压力")
        self.repository.write_cards([original])
        before = self.repository.fingerprint()
        preview = self.repository.preview_import(
            "strain\tn\t压力\n", source_format="paste"
        )

        real_replace = os.replace
        with mock.patch("vocabulary_repository.os.replace", wraps=real_replace) as replace:
            result = self.repository.commit_preview(preview)

        self.assertTrue(result.success)
        replace.assert_called_once()
        self.assertEqual((addition,), result.added_cards)
        self.assertEqual([original, addition], list(result.cards))
        self.assertNotEqual(before, result.fingerprint)
        self.assertEqual([original, addition], list(self.repository.load().cards))
        self.assertEqual([], list(self.path.parent.glob(f".{self.path.name}.*.tmp")))

    def test_stale_preview_rechecks_latest_file_and_does_not_duplicate(self) -> None:
        self.repository.write_cards([Card("abandon", "v", "放弃")])
        preview = self.repository.preview_import(
            "strain\tn\t压力\n", source_format="text"
        )
        self.repository.append_cards([Card("strain", "n", "压力")])

        result = self.repository.commit_preview(preview)

        self.assertTrue(result.success)
        self.assertEqual((), result.added_cards)
        self.assertEqual(2, len(result.cards))
        self.assertIn("stale_preview", issue_codes(result))
        self.assertIn("duplicate_in_library", issue_codes(result))

    def test_append_to_missing_file_creates_standard_utf8_csv(self) -> None:
        result = self.repository.append_cards([Card("abandon", "v", "放弃")])

        self.assertTrue(result.success)
        self.assertEqual("utf-8", self.repository.load().encoding)
        self.assertEqual(["abandon"], [card.word for card in result.cards])
        self.assertNotIn(b"\r\n", self.path.read_bytes())

    def test_append_keeps_distinct_senses_and_skips_exact_duplicate(self) -> None:
        self.repository.write_cards([Card("strain", "n", "品种")])

        result = self.repository.append_cards(
            [Card("STRAIN", "N", "品种"), Card("strain", "n", "压力")]
        )

        self.assertTrue(result.success)
        self.assertEqual(["压力"], [card.meaning for card in result.added_cards])
        self.assertEqual(1, len(result.skipped_duplicates))
        self.assertEqual(2, len(result.cards))

    def test_write_cards_replaces_atomically_and_deduplicates(self) -> None:
        self.repository.write_cards([Card("old", "adj", "旧的")])

        result = self.repository.write_cards(
            [Card("new", "adj", "新的"), Card(" NEW ", "ADJ", "新的")]
        )

        self.assertTrue(result.success)
        self.assertEqual(["new"], [card.word for card in result.cards])
        self.assertEqual(1, len(result.skipped_duplicates))
        self.assertEqual(result.cards, self.repository.load().cards)

    def test_append_refuses_structurally_invalid_existing_file(self) -> None:
        original_bytes = b"word,meaning\nabandon,drop\n"
        self.path.write_bytes(original_bytes)

        result = self.repository.append_cards([Card("new", "adj", "新的")])

        self.assertFalse(result.success)
        self.assertIn("missing_columns", issue_codes(result))
        self.assertEqual(original_bytes, self.path.read_bytes())

    def test_parent_path_failure_is_returned_as_write_issue(self) -> None:
        blocking_file = self.path.parent / "not-a-directory"
        blocking_file.write_text("occupied", encoding="utf-8")
        repository = VocabularyRepository(blocking_file / "vocabularies.csv")

        result = repository.write_cards([Card("new", "adj", "新的")])

        self.assertFalse(result.success)
        self.assertEqual((), result.added_cards)
        self.assertIn("write_error", issue_codes(result))
        self.assertEqual("occupied", blocking_file.read_text(encoding="utf-8"))

    def test_read_only_directory_failure_is_returned_as_write_issue(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        permission_error = PermissionError("read-only directory")

        with mock.patch(
            "vocabulary_repository.tempfile.NamedTemporaryFile",
            side_effect=permission_error,
        ):
            result = self.repository.append_cards([Card("new", "adj", "新的")])

        self.assertFalse(result.success)
        self.assertEqual((), result.added_cards)
        self.assertIn("write_error", issue_codes(result))
        self.assertFalse(self.path.exists())

    def test_row_level_errors_allow_loading_but_refuse_lossy_append(self) -> None:
        original_bytes = "word,pos,meaning\nvalid,n,有效\nbad,n\n".encode("utf-8")
        self.path.write_bytes(original_bytes)

        loaded = self.repository.load()
        self.assertEqual(["valid"], [card.word for card in loaded.cards])
        self.assertIn("invalid_column_count", issue_codes(loaded))

        result = self.repository.append_cards([Card("new", "adj", "新的")])

        self.assertFalse(result.success)
        self.assertEqual((), result.added_cards)
        self.assertIn("invalid_column_count", issue_codes(result))
        self.assertEqual(["valid"], [card.word for card in result.cards])
        self.assertEqual(original_bytes, self.path.read_bytes())

    def test_preview_file_supports_csv_and_txt(self) -> None:
        import_csv = self.path.parent / "incoming.csv"
        import_txt = self.path.parent / "incoming.txt"
        import_csv.write_text("word,pos,meaning\na,n,甲\n", encoding="utf-8")
        import_txt.write_text("b\tn\t乙\n", encoding="utf-8")

        csv_preview = self.repository.preview_file(import_csv)
        txt_preview = self.repository.preview_file(import_txt)

        self.assertEqual(["a"], [card.word for card in csv_preview.cards_to_add])
        self.assertEqual(["b"], [card.word for card in txt_preview.cards_to_add])

    def test_fingerprint_changes_when_library_is_changed_externally(self) -> None:
        self.repository.write_cards([Card("a", "n", "甲")])
        first = self.repository.fingerprint()
        self.path.write_text("word,pos,meaning\nb,n,乙\n", encoding="utf-8")

        second = self.repository.fingerprint()

        self.assertNotEqual(first, second)
        self.assertEqual(["b"], [card.word for card in self.repository.load().cards])


if __name__ == "__main__":
    unittest.main()
