from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import unittest

from domain import Card, Progress, stable_card_id
from scheduler import (
    DEFAULT_BATCH_SIZE,
    build_batch,
    priority_for,
    rank_cards,
    retention_for,
    select_cards,
)


NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)


def reviewed_progress(
    card: Card,
    *,
    mastery: float = 0.8,
    stability_days: float = 2.0,
    elapsed_days: float = 0.0,
) -> Progress:
    reviewed_at = NOW - timedelta(days=elapsed_days)
    return Progress(
        card_id=card.card_id,
        attempts=1,
        mastery=mastery,
        stability_days=stability_days,
        last_reviewed_at=reviewed_at,
        last_score=mastery * 100,
        updated_at=reviewed_at,
    )


class CardIdentityTests(unittest.TestCase):
    def test_id_is_stable_across_harmless_csv_formatting(self) -> None:
        first = Card("  Appeal\t", " N ", "发出   呼吁")
        second = Card("appeal", "n", "发出 呼吁")
        self.assertEqual(first.card_id, second.card_id)
        self.assertEqual(first.card_id, stable_card_id("APPEAL", "n", "发出 呼吁"))

    def test_id_is_stable_under_nfkc_normalization(self) -> None:
        self.assertEqual(Card("ＡＢＣ", "Ｎ", "含义").card_id, Card("abc", "n", "含义").card_id)

    def test_different_senses_have_different_ids(self) -> None:
        noun = Card("appeal", "n", "呼吁")
        verb = Card("appeal", "v", "有吸引力")
        other_noun = Card("appeal", "n", "吸引力")
        self.assertEqual(len({noun.card_id, verb.card_id, other_noun.card_id}), 3)


class SchedulerTests(unittest.TestCase):
    def test_exponential_retention_curve(self) -> None:
        card = Card("word", "n", "meaning")
        progress = reviewed_progress(card, stability_days=2.0)
        self.assertAlmostEqual(retention_for(progress, now=NOW), 1.0)
        self.assertAlmostEqual(
            retention_for(progress, now=NOW + timedelta(days=2)),
            math.exp(-1),
        )
        self.assertAlmostEqual(
            retention_for(progress, now=NOW + timedelta(days=4)),
            math.exp(-2),
        )

    def test_greater_stability_forgets_more_slowly(self) -> None:
        card = Card("word", "n", "meaning")
        weak = reviewed_progress(card, stability_days=1.0)
        strong = reviewed_progress(card, stability_days=10.0)
        future = NOW + timedelta(days=2)
        self.assertLess(retention_for(weak, now=future), retention_for(strong, now=future))

    def test_missing_and_future_timestamps_are_safe(self) -> None:
        missing = Progress(card_id="card", attempts=1, mastery=1, last_reviewed_at=None)
        self.assertEqual(retention_for(missing, now=NOW), 0.0)

        future = Progress(
            card_id="card",
            attempts=1,
            mastery=1,
            stability_days=1,
            last_reviewed_at=NOW + timedelta(days=3),
        )
        self.assertEqual(retention_for(future, now=NOW), 1.0)

    def test_priority_combines_mastery_and_forgetting(self) -> None:
        card = Card("word", "n", "meaning")
        low_mastery = reviewed_progress(card, mastery=0.2, elapsed_days=0)
        high_mastery = reviewed_progress(card, mastery=0.9, elapsed_days=0)
        forgotten = reviewed_progress(card, mastery=0.9, stability_days=1, elapsed_days=10)

        self.assertGreater(priority_for(low_mastery, now=NOW), priority_for(high_mastery, now=NOW))
        self.assertGreater(priority_for(forgotten, now=NOW), priority_for(high_mastery, now=NOW))
        self.assertEqual(priority_for(None, now=NOW), 1.0)

    def test_rank_is_deterministic_and_uses_source_order_for_ties(self) -> None:
        cards = [
            Card("first", "n", "one"),
            Card("second", "n", "two"),
            Card("third", "n", "three"),
        ]
        first = rank_cards(cards, {}, now=NOW)
        second = rank_cards(cards, {}, now=NOW)
        self.assertEqual([item.card for item in first], cards)
        self.assertEqual(first, second)

    def test_rank_deduplicates_exact_cards(self) -> None:
        first = Card("Appeal", "N", "呼吁")
        duplicate = Card(" appeal ", "n", "呼吁")
        other = Card("appeal", "n", "吸引力")
        ranked = rank_cards([first, duplicate, other], {}, now=NOW)
        self.assertEqual([item.card for item in ranked], [first, other])

    def test_default_batch_has_twenty_cards_and_five_reviews(self) -> None:
        new_cards = [Card(f"new-{index}", "n", "meaning") for index in range(30)]
        review_cards = [Card(f"review-{index}", "n", "meaning") for index in range(10)]
        progress = {
            card.card_id: reviewed_progress(card, mastery=1.0, stability_days=10.0)
            for card in review_cards
        }

        batch = build_batch(new_cards + review_cards, progress, now=NOW)
        self.assertEqual(len(batch), DEFAULT_BATCH_SIZE)
        self.assertEqual(sum(item.is_review for item in batch), 5)
        self.assertEqual(len({item.card.card_id for item in batch}), DEFAULT_BATCH_SIZE)
        for start in range(0, DEFAULT_BATCH_SIZE, 4):
            self.assertTrue(
                any(item.is_review for item in batch[start : start + 4]),
                msg=f"missing review in positions {start + 1}-{start + 4}",
            )

    def test_interleaving_preserves_priority_order_inside_each_pool(self) -> None:
        new_cards = [Card(f"new-{index}", "n", "meaning") for index in range(30)]
        review_cards = [Card(f"review-{index}", "n", "meaning") for index in range(8)]
        progress = {
            card.card_id: reviewed_progress(
                card,
                mastery=0.95 - index * 0.05,
                stability_days=2.0,
                elapsed_days=index,
            )
            for index, card in enumerate(review_cards)
        }

        batch = build_batch(new_cards + review_cards, progress, now=NOW)
        selected_review_ids = [item.card.card_id for item in batch if item.is_review]
        selected_new_ids = [item.card.card_id for item in batch if not item.is_review]
        ranked = rank_cards(new_cards + review_cards, progress, now=NOW)
        expected_review_ids = [
            item.card.card_id for item in ranked if item.is_review
        ][: len(selected_review_ids)]
        expected_new_ids = [
            item.card.card_id for item in ranked if not item.is_review
        ][: len(selected_new_ids)]

        self.assertEqual(selected_review_ids, expected_review_ids)
        self.assertEqual(selected_new_ids, expected_new_ids)

    def test_custom_batch_sizes_keep_a_review_in_every_four_card_group(self) -> None:
        new_cards = [Card(f"new-{index}", "n", "meaning") for index in range(120)]
        review_cards = [Card(f"review-{index}", "n", "meaning") for index in range(120)]
        progress = {card.card_id: reviewed_progress(card) for card in review_cards}

        for batch_size in (5, 10, 25, 100):
            with self.subTest(batch_size=batch_size):
                batch = build_batch(
                    new_cards + review_cards,
                    progress,
                    batch_size=batch_size,
                    now=NOW,
                )
                for start in range(0, batch_size, 4):
                    self.assertTrue(any(item.is_review for item in batch[start : start + 4]))

    def test_batch_includes_all_reviews_when_floor_cannot_be_met(self) -> None:
        new_cards = [Card(f"new-{index}", "n", "meaning") for index in range(30)]
        review_cards = [Card(f"review-{index}", "n", "meaning") for index in range(2)]
        progress = {card.card_id: reviewed_progress(card) for card in review_cards}

        batch = build_batch(new_cards + review_cards, progress, now=NOW)
        self.assertEqual(len(batch), DEFAULT_BATCH_SIZE)
        self.assertEqual(sum(item.is_review for item in batch), 2)
        self.assertEqual(
            [index for index, item in enumerate(batch) if item.is_review],
            [9, 19],
        )

    def test_review_floor_uses_actual_capacity_for_small_vocabularies(self) -> None:
        new_cards = [Card(f"new-{index}", "n", "meaning") for index in range(6)]
        review_cards = [Card(f"review-{index}", "n", "meaning") for index in range(2)]
        progress = {card.card_id: reviewed_progress(card) for card in review_cards}

        batch = build_batch(new_cards + review_cards, progress, batch_size=20, now=NOW)
        self.assertEqual(len(batch), 8)
        self.assertEqual(sum(item.is_review for item in batch), 2)

    def test_custom_zero_review_floor_allows_pure_priority_selection(self) -> None:
        new_cards = [Card(f"new-{index}", "n", "meaning") for index in range(5)]
        review = Card("review", "n", "meaning")
        progress = {review.card_id: reviewed_progress(review, mastery=1.0)}
        batch = build_batch(
            new_cards + [review],
            progress,
            batch_size=3,
            min_review_fraction=0,
            now=NOW,
        )
        self.assertEqual(sum(item.is_review for item in batch), 0)

    def test_select_cards_returns_plain_card_queue(self) -> None:
        cards = [Card(f"word-{index}", "n", "meaning") for index in range(3)]
        self.assertEqual(select_cards(cards, {}, batch_size=2, now=NOW), cards[:2])

    def test_nonpositive_batch_size_is_empty(self) -> None:
        cards = [Card("word", "n", "meaning")]
        self.assertEqual(build_batch(cards, {}, batch_size=0, now=NOW), [])


if __name__ == "__main__":
    unittest.main()
