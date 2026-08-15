"""Core domain types and learning-state update rules.

The module deliberately has no Streamlit or persistence dependencies so the
same card identity and progress rules can be reused by imports, scheduling,
and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any
import unicodedata


UTC = timezone.utc
CARD_ID_VERSION = "card-v1"
DEFAULT_STABILITY_DAYS = 1.0
MIN_STABILITY_DAYS = 0.25
MAX_STABILITY_DAYS = 365.0
MASTERY_HISTORY_WEIGHT = 0.70
MASTERY_SCORE_WEIGHT = 0.30

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_card_field(value: Any, *, casefold: bool = False) -> str:
    """Return the canonical representation used to identify a card.

    Display text remains untouched on :class:`Card`; normalization is only for
    identity. NFKC handles common full-width/compatibility variants, while
    whitespace folding keeps IDs stable after harmless CSV formatting edits.
    """

    normalized = unicodedata.normalize("NFKC", "" if value is None else str(value))
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold() if casefold else normalized


def stable_card_id(word: Any, pos: Any, meaning: Any) -> str:
    """Build a deterministic, sense-specific ID from the existing CSV fields."""

    payload = json.dumps(
        [
            normalize_card_field(word, casefold=True),
            normalize_card_field(pos, casefold=True),
            normalize_card_field(meaning),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{CARD_ID_VERSION}:{digest}"


def utc_now() -> datetime:
    """Return an aware UTC timestamp (kept as a function for test injection)."""

    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC; legacy naive values are interpreted as UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_utc(value: Any) -> datetime | None:
    """Parse a stored timestamp defensively, returning ``None`` when invalid."""

    if isinstance(value, datetime):
        return ensure_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        return ensure_utc(datetime.fromisoformat(candidate))
    except (TypeError, ValueError, OverflowError):
        return None


def format_utc(value: datetime) -> str:
    """Serialize an aware/naive datetime as an ISO-8601 UTC string."""

    return ensure_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _finite_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return converted if math.isfinite(converted) else default


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0, converted)


@dataclass(frozen=True)
class Card:
    """A vocabulary sense backed by the current ``word,pos,meaning`` CSV."""

    word: str
    pos: str
    meaning: str

    @property
    def card_id(self) -> str:
        return stable_card_id(self.word, self.pos, self.meaning)


@dataclass(frozen=True)
class Progress:
    """The current scheduling state for one card."""

    card_id: str
    attempts: int = 0
    mastery: float = 0.0
    stability_days: float = DEFAULT_STABILITY_DAYS
    last_reviewed_at: datetime | None = None
    last_score: float | None = None
    updated_at: datetime | None = None

    @classmethod
    def sanitized(
        cls,
        *,
        card_id: Any,
        attempts: Any = 0,
        mastery: Any = 0.0,
        stability_days: Any = DEFAULT_STABILITY_DAYS,
        last_reviewed_at: Any = None,
        last_score: Any = None,
        updated_at: Any = None,
    ) -> "Progress":
        """Create safe domain state from possibly damaged SQLite values."""

        safe_stability = _finite_float(stability_days, DEFAULT_STABILITY_DAYS)
        if safe_stability <= 0:
            safe_stability = DEFAULT_STABILITY_DAYS
        safe_stability = min(MAX_STABILITY_DAYS, max(MIN_STABILITY_DAYS, safe_stability))

        safe_score: float | None
        if last_score is None:
            safe_score = None
        else:
            candidate = _finite_float(last_score, math.nan)
            safe_score = None if not math.isfinite(candidate) else min(100.0, max(0.0, candidate))

        return cls(
            card_id=str(card_id or "").strip(),
            attempts=_nonnegative_int(attempts),
            mastery=min(1.0, max(0.0, _finite_float(mastery, 0.0))),
            stability_days=safe_stability,
            last_reviewed_at=parse_utc(last_reviewed_at),
            last_score=safe_score,
            updated_at=parse_utc(updated_at),
        )


@dataclass(frozen=True)
class ReviewEvent:
    """An immutable historical review event."""

    event_id: int | None
    card_id: str
    word: str
    pos: str
    meaning: str
    reviewed_at: datetime
    score: float
    status: str
    mastery_before: float
    mastery_after: float
    stability_before: float
    stability_after: float
    sentence: str | None = None
    reference_translation: str | None = None
    user_translation: str | None = None
    feedback: str | None = None
    review_key: str | None = None

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"


@dataclass(frozen=True)
class ScheduledCard:
    """A card plus the scheduling values calculated for the current instant."""

    card: Card
    priority: float
    retention: float
    is_review: bool


def evolve_progress(
    card_id: str,
    previous: Progress | None,
    score: float,
    *,
    reviewed_at: datetime | None = None,
) -> Progress:
    """Apply one successful evaluation (or an explicit skip scored as zero).

    Mastery is an EMA after the first attempt. Stability shrinks after a score
    below 60 and expands after a passing score, bounded to 6 hours..365 days.
    """

    if not str(card_id).strip():
        raise ValueError("card_id must not be empty")

    safe_score = min(100.0, max(0.0, _finite_float(score, 0.0)))
    quality = safe_score / 100.0
    timestamp = ensure_utc(reviewed_at or utc_now())
    old = previous or Progress(card_id=card_id)
    old = Progress.sanitized(
        card_id=card_id,
        attempts=old.attempts,
        mastery=old.mastery,
        stability_days=old.stability_days,
        last_reviewed_at=old.last_reviewed_at,
        last_score=old.last_score,
        updated_at=old.updated_at,
    )

    if old.attempts == 0:
        mastery = quality
    else:
        mastery = (
            MASTERY_HISTORY_WEIGHT * old.mastery
            + MASTERY_SCORE_WEIGHT * quality
        )

    if old.attempts == 0:
        if safe_score < 60.0:
            stability = 0.25
        elif safe_score < 80.0:
            stability = 1.0
        elif safe_score < 90.0:
            stability = 3.0
        else:
            stability = 7.0
    else:
        if safe_score < 60.0:
            stability_multiplier = 0.50
        elif safe_score < 80.0:
            stability_multiplier = 1.20
        elif safe_score < 90.0:
            stability_multiplier = 2.0
        else:
            stability_multiplier = 2.50
        stability = old.stability_days * stability_multiplier
        stability = min(MAX_STABILITY_DAYS, max(MIN_STABILITY_DAYS, stability))

    return Progress(
        card_id=card_id,
        attempts=old.attempts + 1,
        mastery=min(1.0, max(0.0, mastery)),
        stability_days=stability,
        last_reviewed_at=timestamp,
        last_score=safe_score,
        updated_at=timestamp,
    )
