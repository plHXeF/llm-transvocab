"""Deterministic mastery- and forgetting-aware vocabulary scheduling."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import math
from typing import Iterable, Mapping

from domain import (
    DEFAULT_STABILITY_DAYS,
    Card,
    Progress,
    ScheduledCard,
    ensure_utc,
    utc_now,
)


DEFAULT_BATCH_SIZE = 20
MIN_REVIEW_FRACTION = 0.25


def retention_for(progress: Progress | None, *, now: datetime | None = None) -> float:
    """Return Ebbinghaus retention ``exp(-elapsed / stability)`` in ``[0, 1]``.

    A missing/never-reviewed timestamp is deliberately treated as no retained
    memory. Future timestamps are clamped to zero elapsed time so clock skew
    cannot produce retention above one.
    """

    if progress is None or progress.attempts <= 0 or progress.last_reviewed_at is None:
        return 0.0

    current = ensure_utc(now or utc_now())
    reviewed = ensure_utc(progress.last_reviewed_at)
    elapsed_days = max(0.0, (current - reviewed).total_seconds() / 86_400.0)
    stability = progress.stability_days
    if not isinstance(stability, (int, float)) or not math.isfinite(stability) or stability <= 0:
        stability = DEFAULT_STABILITY_DAYS
    return min(1.0, max(0.0, math.exp(-elapsed_days / float(stability))))


def priority_for(progress: Progress | None, *, now: datetime | None = None) -> float:
    """Return review urgency as one minus predicted effective mastery."""

    if progress is None or progress.attempts <= 0:
        return 1.0
    mastery = progress.mastery
    if not isinstance(mastery, (int, float)) or not math.isfinite(mastery):
        mastery = 0.0
    mastery = min(1.0, max(0.0, float(mastery)))
    effective_mastery = mastery * retention_for(progress, now=now)
    return min(1.0, max(0.0, 1.0 - effective_mastery))


def rank_cards(
    cards: Iterable[Card],
    progress_by_card_id: Mapping[str, Progress] | None = None,
    *,
    now: datetime | None = None,
) -> list[ScheduledCard]:
    """Rank unique active cards deterministically by descending urgency.

    The first occurrence wins when an imported vocabulary accidentally
    contains an exact duplicate. CSV order is the deterministic tie-breaker.
    """

    current = ensure_utc(now or utc_now())
    progress_map = progress_by_card_id or {}
    ranked_with_order: list[tuple[int, ScheduledCard]] = []
    seen: set[str] = set()

    for source_order, card in enumerate(cards):
        if card.card_id in seen:
            continue
        seen.add(card.card_id)
        progress = progress_map.get(card.card_id)
        is_review = bool(progress is not None and progress.attempts > 0)
        ranked_with_order.append(
            (
                source_order,
                ScheduledCard(
                    card=card,
                    priority=priority_for(progress, now=current),
                    retention=retention_for(progress, now=current),
                    is_review=is_review,
                ),
            )
        )

    ranked_with_order.sort(key=lambda item: (-item[1].priority, item[0]))
    return [scheduled for _, scheduled in ranked_with_order]


def build_batch(
    cards: Iterable[Card],
    progress_by_card_id: Mapping[str, Progress] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_review_fraction: float = MIN_REVIEW_FRACTION,
    now: datetime | None = None,
) -> list[ScheduledCard]:
    """Build a deterministic batch with a review-card floor.

    When enough previously attempted cards exist, at least ``ceil(25%)`` of
    the selected batch are reviews. If fewer reviews exist, all are included.
    The returned cards retain global priority order after composition.
    """

    if batch_size <= 0:
        return []
    safe_fraction = min(1.0, max(0.0, float(min_review_fraction)))
    ranked = rank_cards(cards, progress_by_card_id, now=now)
    capacity = min(int(batch_size), len(ranked))
    if capacity <= 0:
        return []

    review_candidates = [item for item in ranked if item.is_review]
    review_target = min(len(review_candidates), math.ceil(capacity * safe_fraction))
    forced_ids = {item.card.card_id for item in review_candidates[:review_target]}

    selected_ids = set(forced_ids)
    for item in ranked:
        if len(selected_ids) >= capacity:
            break
        selected_ids.add(item.card.card_id)

    selected = [item for item in ranked if item.card.card_id in selected_ids][:capacity]
    if review_target <= 0:
        return selected

    # Spread the guaranteed reviews throughout the batch instead of leaving
    # them at the end behind new cards (whose default priority is one). With a
    # 25% floor this produces slots 4, 8, 12, ... for a 20-card batch. For a
    # partial/scarce batch it spaces the available reviews as evenly as possible.
    review_slots = {
        math.ceil((index + 1) * capacity / review_target) - 1
        for index in range(review_target)
    }
    global_order = {
        item.card.card_id: index for index, item in enumerate(selected)
    }
    reviews = deque(item for item in selected if item.is_review)
    others = deque(item for item in selected if not item.is_review)
    interleaved: list[ScheduledCard] = []

    for position in range(capacity):
        future_review_slots = sum(slot > position for slot in review_slots)
        if position in review_slots and reviews:
            interleaved.append(reviews.popleft())
        elif not reviews:
            interleaved.append(others.popleft())
        elif not others:
            interleaved.append(reviews.popleft())
        elif len(reviews) <= future_review_slots:
            # Keep enough review cards for the remaining guaranteed slots.
            interleaved.append(others.popleft())
        elif global_order[reviews[0].card.card_id] < global_order[others[0].card.card_id]:
            interleaved.append(reviews.popleft())
        else:
            interleaved.append(others.popleft())

    return interleaved


def select_cards(
    cards: Iterable[Card],
    progress_by_card_id: Mapping[str, Progress] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_review_fraction: float = MIN_REVIEW_FRACTION,
    now: datetime | None = None,
) -> list[Card]:
    """Convenience wrapper returning only cards for the learning-session queue."""

    return [
        item.card
        for item in build_batch(
            cards,
            progress_by_card_id,
            batch_size=batch_size,
            min_review_fraction=min_review_fraction,
            now=now,
        )
    ]
