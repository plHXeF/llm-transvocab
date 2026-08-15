from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
import warnings

from domain import Card
from learning_store import LearningStore, SCHEMA_VERSION


class LearningStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "learning.db"
        self.store = LearningStore(self.db_path)
        self.card = Card("appeal", "n", "呼吁")
        self.reviewed_at = datetime(2026, 8, 15, 8, 30, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_schema_and_meta_are_initialized(self) -> None:
        self.assertEqual(self.store.schema_version(), SCHEMA_VERSION)
        self.assertIsNotNone(self.store.get_meta("created_at"))
        self.assertEqual(self.store.load_progress(), {})
        self.assertEqual(self.store.list_review_events(), [])

    def test_record_review_persists_progress_and_full_event_snapshot(self) -> None:
        progress = self.store.record_review(
            self.card,
            85,
            reviewed_at=self.reviewed_at,
            sentence="They made an appeal for calm.",
            reference_translation="他们呼吁保持冷静。",
            user_translation="他们发出保持冷静的呼吁。",
            feedback="准确。",
        )

        self.assertEqual(progress.attempts, 1)
        self.assertAlmostEqual(progress.mastery, 0.85)
        self.assertAlmostEqual(progress.stability_days, 3.0)
        self.assertEqual(progress.last_reviewed_at, self.reviewed_at)

        reopened = LearningStore(self.db_path)
        persisted = reopened.get_progress(self.card.card_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted, progress)

        events = reopened.list_review_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.card_id, self.card.card_id)
        self.assertEqual((event.word, event.pos, event.meaning), ("appeal", "n", "呼吁"))
        self.assertEqual(event.status, "answered")
        self.assertFalse(event.skipped)
        self.assertEqual(event.sentence, "They made an appeal for calm.")
        self.assertEqual(event.reference_translation, "他们呼吁保持冷静。")
        self.assertEqual(event.user_translation, "他们发出保持冷静的呼吁。")
        self.assertEqual(event.feedback, "准确。")
        self.assertIsNone(event.review_key)
        self.assertEqual(event.reviewed_at, self.reviewed_at)
        self.assertAlmostEqual(event.mastery_before, 0.0)
        self.assertAlmostEqual(event.mastery_after, 0.85)

    def test_first_stability_uses_score_bands(self) -> None:
        cases = ((59, 0.25), (60, 1.0), (79.99, 1.0), (80, 3.0), (89.99, 3.0), (90, 7.0))
        for index, (score, expected) in enumerate(cases):
            card = Card(f"word-{index}", "n", "meaning")
            progress = self.store.record_review(card, score, reviewed_at=self.reviewed_at)
            self.assertAlmostEqual(progress.stability_days, expected, msg=f"score={score}")

    def test_subsequent_reviews_use_ema_and_stability_multipliers(self) -> None:
        first = self.store.record_review(self.card, 90, reviewed_at=self.reviewed_at)
        second = self.store.record_review(
            self.card,
            50,
            reviewed_at=self.reviewed_at + timedelta(days=1),
        )
        self.assertAlmostEqual(first.stability_days, 7.0)
        self.assertAlmostEqual(second.mastery, 0.70 * 0.90 + 0.30 * 0.50)
        self.assertAlmostEqual(second.stability_days, 3.5)

        third = self.store.record_review(
            self.card,
            75,
            reviewed_at=self.reviewed_at + timedelta(days=2),
        )
        self.assertAlmostEqual(third.stability_days, 3.5 * 1.2)
        fourth = self.store.record_review(
            self.card,
            85,
            reviewed_at=self.reviewed_at + timedelta(days=3),
        )
        self.assertAlmostEqual(fourth.stability_days, 3.5 * 1.2 * 2.0)
        fifth = self.store.record_review(
            self.card,
            95,
            reviewed_at=self.reviewed_at + timedelta(days=4),
        )
        self.assertAlmostEqual(fifth.stability_days, 3.5 * 1.2 * 2.0 * 2.5)

    def test_stability_is_clamped_to_six_hours_and_one_year(self) -> None:
        weak = Card("weak", "adj", "薄弱的")
        weak_progress = self.store.record_review(weak, 10, reviewed_at=self.reviewed_at)
        for day in range(1, 5):
            weak_progress = self.store.record_review(
                weak,
                10,
                reviewed_at=self.reviewed_at + timedelta(days=day),
            )
        self.assertEqual(weak_progress.stability_days, 0.25)

        strong = Card("strong", "adj", "强的")
        strong_progress = self.store.record_review(strong, 100, reviewed_at=self.reviewed_at)
        for day in range(1, 10):
            strong_progress = self.store.record_review(
                strong,
                100,
                reviewed_at=self.reviewed_at + timedelta(days=day),
            )
        self.assertEqual(strong_progress.stability_days, 365.0)

    def test_skip_is_zero_score_review(self) -> None:
        progress = self.store.record_review(
            self.card,
            None,
            status="skipped",
            reviewed_at=self.reviewed_at,
        )
        self.assertEqual(progress.attempts, 1)
        self.assertEqual(progress.mastery, 0.0)
        self.assertEqual(progress.last_score, 0.0)
        self.assertEqual(progress.stability_days, 0.25)
        event = self.store.list_review_events()[0]
        self.assertTrue(event.skipped)
        self.assertEqual(event.status, "skipped")

    def test_answered_review_key_replay_is_idempotent(self) -> None:
        first = self.store.record_review(
            self.card,
            90,
            reviewed_at=self.reviewed_at,
            review_key="batch-1:0:appeal",
            feedback="first",
        )
        replay = self.store.record_review(
            self.card,
            10,
            reviewed_at=self.reviewed_at + timedelta(days=7),
            review_key="batch-1:0:appeal",
            feedback="must not replace the first event",
        )

        self.assertEqual(replay, first)
        self.assertEqual(self.store.get_progress(self.card.card_id).attempts, 1)
        events = self.store.list_review_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].score, 90)
        self.assertEqual(events[0].feedback, "first")
        self.assertEqual(events[0].review_key, "batch-1:0:appeal")

    def test_skipped_review_key_replay_is_idempotent(self) -> None:
        first = self.store.record_review(
            self.card,
            None,
            status="skipped",
            reviewed_at=self.reviewed_at,
            review_key="batch-1:0:skip",
        )
        replay = self.store.record_review(
            self.card,
            None,
            status="skipped",
            reviewed_at=self.reviewed_at + timedelta(days=1),
            review_key="batch-1:0:skip",
        )

        self.assertEqual(replay, first)
        self.assertEqual(self.store.get_progress(self.card.card_id).attempts, 1)
        self.assertEqual(len(self.store.list_review_events()), 1)

    def test_review_key_cannot_be_reused_for_another_card(self) -> None:
        self.store.record_review(self.card, 90, review_key="one-action")
        other = Card("genre", "n", "体裁")
        with self.assertRaises(ValueError):
            self.store.record_review(other, 90, review_key="one-action")
        self.assertIsNone(self.store.get_progress(other.card_id))
        self.assertEqual(len(self.store.list_review_events()), 1)

    def test_failed_or_invalid_evaluation_does_not_write_progress(self) -> None:
        for invalid in (None, "not-a-score", float("nan"), float("inf")):
            with self.subTest(score=invalid):
                with self.assertRaises(ValueError):
                    self.store.record_review(self.card, invalid, status="answered")
        self.assertIsNone(self.store.get_progress(self.card.card_id))
        self.assertEqual(self.store.list_review_events(), [])

    def test_reset_progress_preserves_history(self) -> None:
        self.store.record_review(self.card, 90, reviewed_at=self.reviewed_at)
        self.assertEqual(self.store.reset_progress(), 1)
        self.assertIsNone(self.store.get_progress(self.card.card_id))
        self.assertEqual(len(self.store.list_review_events()), 1)
        self.assertIsNotNone(self.store.get_meta("progress_reset_at"))

    def test_clear_history_preserves_current_progress(self) -> None:
        expected = self.store.record_review(self.card, 90, reviewed_at=self.reviewed_at)
        self.assertEqual(self.store.clear_history(), 1)
        self.assertEqual(self.store.get_progress(self.card.card_id), expected)
        self.assertEqual(self.store.list_review_events(), [])
        self.assertIsNotNone(self.store.get_meta("history_cleared_at"))

    def test_clear_all_removes_progress_and_history_but_not_schema(self) -> None:
        self.store.record_review(self.card, 90, reviewed_at=self.reviewed_at)
        deleted = self.store.clear_all()
        self.assertEqual(deleted, {"progress": 1, "review_events": 1})
        self.assertEqual(self.store.load_progress(), {})
        self.assertEqual(self.store.list_review_events(), [])
        self.assertEqual(self.store.schema_version(), SCHEMA_VERSION)

    def test_active_filter_ignores_orphaned_progress_without_deleting_it(self) -> None:
        other = Card("appeal", "v", "有吸引力")
        self.store.record_review(self.card, 90, reviewed_at=self.reviewed_at)
        self.store.record_review(other, 70, reviewed_at=self.reviewed_at)

        active = self.store.load_progress([other.card_id])
        self.assertEqual(set(active), {other.card_id})
        self.assertIsNotNone(self.store.get_progress(self.card.card_id))

    def test_malformed_progress_values_are_sanitized(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO progress(
                    card_id, attempts, mastery, stability_days,
                    last_reviewed_at, last_score, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("damaged", -9, "bad", -10, "not-a-time", 999, "also-bad"),
            )

        progress = self.store.get_progress("damaged")
        self.assertIsNotNone(progress)
        self.assertEqual(progress.attempts, 0)
        self.assertEqual(progress.mastery, 0.0)
        self.assertEqual(progress.stability_days, 1.0)
        self.assertIsNone(progress.last_reviewed_at)
        self.assertEqual(progress.last_score, 100.0)
        self.assertIsNone(progress.updated_at)

    def test_naive_and_non_utc_timestamps_are_saved_as_utc(self) -> None:
        naive = datetime(2026, 8, 15, 8, 30)
        first = self.store.record_review(self.card, 90, reviewed_at=naive)
        self.assertEqual(first.last_reviewed_at.tzinfo, timezone.utc)
        self.assertEqual(first.last_reviewed_at.hour, 8)

        second_card = Card("genre", "n", "体裁")
        china_time = datetime(2026, 8, 15, 16, 30, tzinfo=timezone(timedelta(hours=8)))
        second = self.store.record_review(second_card, 90, reviewed_at=china_time)
        self.assertEqual(second.last_reviewed_at, self.reviewed_at)

    def test_corrupt_database_is_backed_up_and_recreated(self) -> None:
        corrupt_path = Path(self.temp_dir.name) / "corrupt.db"
        corrupt_path.write_bytes(b"this is not sqlite")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recovered = LearningStore(corrupt_path)

        self.assertEqual(recovered.schema_version(), SCHEMA_VERSION)
        self.assertEqual(recovered.load_progress(), {})
        self.assertIsNotNone(recovered.last_recovery_backup)
        self.assertTrue(recovered.last_recovery_backup.exists())
        self.assertTrue(any("corrupt learning database" in str(item.message) for item in caught))

    def test_v1_database_migrates_without_losing_progress_or_history(self) -> None:
        legacy_path = Path(self.temp_dir.name) / "legacy-v1.db"
        reviewed_at = "2026-08-15T08:30:00.000000Z"
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE progress (
                    card_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL,
                    mastery REAL NOT NULL,
                    stability_days REAL NOT NULL,
                    last_reviewed_at TEXT,
                    last_score REAL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    pos TEXT NOT NULL,
                    meaning TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    score REAL NOT NULL,
                    status TEXT NOT NULL,
                    mastery_before REAL NOT NULL,
                    mastery_after REAL NOT NULL,
                    stability_before REAL NOT NULL,
                    stability_after REAL NOT NULL,
                    sentence TEXT,
                    reference_translation TEXT,
                    user_translation TEXT,
                    feedback TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', '1')"
            )
            connection.execute(
                """
                INSERT INTO progress(
                    card_id, attempts, mastery, stability_days,
                    last_reviewed_at, last_score, updated_at
                ) VALUES(?, 1, 0.9, 7.0, ?, 90.0, ?)
                """,
                (self.card.card_id, reviewed_at, reviewed_at),
            )
            connection.execute(
                """
                INSERT INTO review_events(
                    card_id, word, pos, meaning, reviewed_at, score, status,
                    mastery_before, mastery_after, stability_before, stability_after,
                    sentence, reference_translation, user_translation, feedback
                ) VALUES(?, ?, ?, ?, ?, 90, 'answered', 0, .9, 1, 7, NULL, NULL, NULL, 'legacy')
                """,
                (
                    self.card.card_id,
                    self.card.word,
                    self.card.pos,
                    self.card.meaning,
                    reviewed_at,
                ),
            )

        migrated = LearningStore(legacy_path)
        self.assertEqual(migrated.schema_version(), 2)
        self.assertEqual(migrated.get_progress(self.card.card_id).attempts, 1)
        events = migrated.list_review_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].feedback, "legacy")
        self.assertIsNone(events[0].review_key)
        with sqlite3.connect(legacy_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(review_events)")
            }
            indexes = {
                row[1] for row in connection.execute("PRAGMA index_list(review_events)")
            }
        self.assertIn("review_key", columns)
        self.assertIn("idx_review_events_review_key", indexes)

        updated = migrated.record_review(
            self.card,
            80,
            review_key="post-migration-review",
            reviewed_at=self.reviewed_at + timedelta(days=1),
        )
        self.assertEqual(updated.attempts, 2)
        self.assertEqual(len(migrated.list_review_events()), 2)


if __name__ == "__main__":
    unittest.main()
