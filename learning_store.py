"""SQLite persistence for current learning state and immutable review history."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Iterable, TypeVar
import warnings

from domain import (
    DEFAULT_STABILITY_DAYS,
    Card,
    Progress,
    ReviewEvent,
    evolve_progress,
    format_utc,
    parse_utc,
    utc_now,
)


SCHEMA_VERSION = 2
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "learning.db"
VALID_REVIEW_STATUSES = frozenset({"answered", "skipped"})

_T = TypeVar("_T")


class UnsupportedSchemaError(RuntimeError):
    """Raised when a newer application schema owns the database."""


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


class LearningStore:
    """Persist progress and review events in a small, versioned SQLite DB.

    Connections are short-lived and writes use SQLite transactions. This fits
    Streamlit's rerun model and avoids keeping thread-affine connection objects
    in session state.
    """

    def __init__(self, db_path: str | os.PathLike[str] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.last_recovery_backup: Path | None = None
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_with_recovery(self._initialize_schema)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS progress (
                    card_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL,
                    mastery REAL NOT NULL,
                    stability_days REAL NOT NULL,
                    last_reviewed_at TEXT,
                    last_score REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_key TEXT UNIQUE,
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

                CREATE INDEX IF NOT EXISTS idx_review_events_card_time
                    ON review_events(card_id, reviewed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_review_events_time
                    ON review_events(reviewed_at DESC);
                """
            )

            version_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if version_row is None:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?)",
                    (format_utc(utc_now()),),
                )
            else:
                try:
                    stored_version = int(version_row["value"])
                except (TypeError, ValueError, OverflowError):
                    stored_version = 0
                if stored_version > SCHEMA_VERSION:
                    raise UnsupportedSchemaError(
                        f"database schema {stored_version} is newer than supported {SCHEMA_VERSION}"
                    )
                if stored_version < SCHEMA_VERSION:
                    self._migrate(connection, stored_version)

            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_review_events_review_key "
                "ON review_events(review_key) WHERE review_key IS NOT NULL"
            )

            check_row = connection.execute("PRAGMA quick_check").fetchone()
            if check_row is None or check_row[0] != "ok":
                detail = "unknown corruption" if check_row is None else check_row[0]
                raise sqlite3.DatabaseError(f"database disk image is malformed: {detail}")

    def _migrate(self, connection: sqlite3.Connection, stored_version: int) -> None:
        """Migrate older schemas in-place without discarding review history."""

        version = max(0, stored_version)
        if version < 2:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(review_events)").fetchall()
            }
            if "review_key" not in columns:
                connection.execute("ALTER TABLE review_events ADD COLUMN review_key TEXT")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_review_events_review_key "
                "ON review_events(review_key) WHERE review_key IS NOT NULL"
            )
            version = 2

        connection.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(version),),
        )

    @staticmethod
    def _is_corruption_error(error: sqlite3.DatabaseError) -> bool:
        message = str(error).casefold()
        return any(
            marker in message
            for marker in (
                "database disk image is malformed",
                "file is not a database",
                "database corruption",
                "malformed database schema",
            )
        )

    def _run_with_recovery(self, operation: Callable[[], _T]) -> _T:
        with self._lock:
            try:
                return operation()
            except sqlite3.DatabaseError as error:
                if not self._is_corruption_error(error):
                    raise
                self._recover_corrupt_database(error)
                self._initialize_schema()
                return operation()

    def _recover_corrupt_database(self, error: sqlite3.DatabaseError) -> None:
        stamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.db_path.with_name(f"{self.db_path.name}.corrupt-{stamp}")
        if self.db_path.exists():
            os.replace(self.db_path, backup)
            self.last_recovery_backup = backup
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.db_path}{suffix}")
            if sidecar.exists():
                os.replace(sidecar, Path(f"{backup}{suffix}"))
        warnings.warn(
            f"Recovered a corrupt learning database; original saved as {backup}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _row_to_progress(row: sqlite3.Row) -> Progress:
        return Progress.sanitized(
            card_id=row["card_id"],
            attempts=row["attempts"],
            mastery=row["mastery"],
            stability_days=row["stability_days"],
            last_reviewed_at=row["last_reviewed_at"],
            last_score=row["last_score"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ReviewEvent:
        reviewed_at = parse_utc(row["reviewed_at"]) or datetime(1970, 1, 1, tzinfo=timezone.utc)
        status = _safe_text(row["status"], "answered")
        if status not in VALID_REVIEW_STATUSES:
            status = "answered"
        return ReviewEvent(
            event_id=max(0, int(_safe_float(row["event_id"], 0.0))) or None,
            card_id=_safe_text(row["card_id"]).strip(),
            word=_safe_text(row["word"]),
            pos=_safe_text(row["pos"]),
            meaning=_safe_text(row["meaning"]),
            reviewed_at=reviewed_at,
            score=min(100.0, max(0.0, _safe_float(row["score"]))),
            status=status,
            mastery_before=min(1.0, max(0.0, _safe_float(row["mastery_before"]))),
            mastery_after=min(1.0, max(0.0, _safe_float(row["mastery_after"]))),
            stability_before=max(0.0, _safe_float(row["stability_before"])),
            stability_after=max(0.0, _safe_float(row["stability_after"])),
            sentence=row["sentence"] if isinstance(row["sentence"], str) else None,
            reference_translation=(
                row["reference_translation"]
                if isinstance(row["reference_translation"], str)
                else None
            ),
            user_translation=(
                row["user_translation"] if isinstance(row["user_translation"], str) else None
            ),
            feedback=row["feedback"] if isinstance(row["feedback"], str) else None,
            review_key=(
                row["review_key"] if isinstance(row["review_key"], str) else None
            ),
        )

    def schema_version(self) -> int:
        value = self.get_meta("schema_version")
        try:
            return int(value) if value is not None else SCHEMA_VERSION
        except (TypeError, ValueError, OverflowError):
            return SCHEMA_VERSION

    def get_meta(self, key: str) -> str | None:
        def operation() -> str | None:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT value FROM meta WHERE key = ?", (str(key),)
                ).fetchone()
            return None if row is None else _safe_text(row["value"])

        return self._run_with_recovery(operation)

    def set_meta(self, key: str, value: Any) -> None:
        if not str(key).strip():
            raise ValueError("meta key must not be empty")

        def operation() -> None:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(key), _safe_text(value)),
                )

        self._run_with_recovery(operation)

    def get_progress(self, card_id: str) -> Progress | None:
        def operation() -> Progress | None:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM progress WHERE card_id = ?", (str(card_id),)
                ).fetchone()
            return None if row is None else self._row_to_progress(row)

        return self._run_with_recovery(operation)

    def load_progress(self, active_card_ids: Iterable[str] | None = None) -> dict[str, Progress]:
        """Load sanitized progress, optionally filtering to currently active cards."""

        active = None if active_card_ids is None else {str(item) for item in active_card_ids}
        if active is not None and not active:
            return {}

        def operation() -> dict[str, Progress]:
            with closing(self._connect()) as connection:
                rows = connection.execute("SELECT * FROM progress").fetchall()
            result: dict[str, Progress] = {}
            for row in rows:
                progress = self._row_to_progress(row)
                if not progress.card_id or (active is not None and progress.card_id not in active):
                    continue
                result[progress.card_id] = progress
            return result

        return self._run_with_recovery(operation)

    def list_progress(self, active_card_ids: Iterable[str] | None = None) -> dict[str, Progress]:
        return self.load_progress(active_card_ids)

    def record_review(
        self,
        card: Card | str,
        score: float | None,
        *,
        status: str = "answered",
        reviewed_at: datetime | None = None,
        sentence: str | None = None,
        reference_translation: str | None = None,
        user_translation: str | None = None,
        feedback: str | None = None,
        review_key: str | None = None,
        word: str = "",
        pos: str = "",
        meaning: str = "",
    ) -> Progress:
        """Atomically update progress and append a detailed review event.

        Passing a :class:`Card` is preferred because it snapshots the displayed
        vocabulary fields into history. A raw card ID remains supported for
        callers that do not have the active vocabulary object. Callers should
        provide a stable ``review_key`` for each presented question; replaying
        that key returns the card's current progress without applying the
        review or appending history again.
        """

        if isinstance(card, Card):
            card_id = card.card_id
            snapshot_word, snapshot_pos, snapshot_meaning = card.word, card.pos, card.meaning
        else:
            card_id = str(card).strip()
            snapshot_word, snapshot_pos, snapshot_meaning = word, pos, meaning
        if not card_id:
            raise ValueError("card/card_id must not be empty")
        if status not in VALID_REVIEW_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_REVIEW_STATUSES)}")
        if status == "skipped":
            normalized_score = 0.0
        else:
            if score is None:
                raise ValueError("answered reviews require a score")
            try:
                normalized_score = float(score)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("score must be a finite number") from error
            if not math.isfinite(normalized_score):
                raise ValueError("score must be a finite number")
            normalized_score = min(100.0, max(0.0, normalized_score))
        normalized_review_key = None
        if review_key is not None:
            normalized_review_key = str(review_key).strip() or None
        timestamp = reviewed_at or utc_now()

        def operation() -> Progress:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                if normalized_review_key is not None:
                    replay = connection.execute(
                        "SELECT card_id FROM review_events WHERE review_key = ?",
                        (normalized_review_key,),
                    ).fetchone()
                    if replay is not None:
                        replay_card_id = _safe_text(replay["card_id"]).strip()
                        if replay_card_id != card_id:
                            raise ValueError(
                                "review_key is already associated with another card"
                            )
                        current_row = connection.execute(
                            "SELECT * FROM progress WHERE card_id = ?", (card_id,)
                        ).fetchone()
                        if current_row is None:
                            return Progress(card_id=card_id)
                        return self._row_to_progress(current_row)
                row = connection.execute(
                    "SELECT * FROM progress WHERE card_id = ?", (card_id,)
                ).fetchone()
                before = None if row is None else self._row_to_progress(row)
                after = evolve_progress(
                    card_id,
                    before,
                    normalized_score,
                    reviewed_at=timestamp,
                )
                connection.execute(
                    """
                    INSERT INTO progress(
                        card_id, attempts, mastery, stability_days,
                        last_reviewed_at, last_score, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(card_id) DO UPDATE SET
                        attempts = excluded.attempts,
                        mastery = excluded.mastery,
                        stability_days = excluded.stability_days,
                        last_reviewed_at = excluded.last_reviewed_at,
                        last_score = excluded.last_score,
                        updated_at = excluded.updated_at
                    """,
                    (
                        after.card_id,
                        after.attempts,
                        after.mastery,
                        after.stability_days,
                        format_utc(after.last_reviewed_at),
                        after.last_score,
                        format_utc(after.updated_at),
                    ),
                )
                before_mastery = 0.0 if before is None else before.mastery
                before_stability = DEFAULT_STABILITY_DAYS if before is None else before.stability_days
                connection.execute(
                    """
                    INSERT INTO review_events(
                        review_key, card_id, word, pos, meaning, reviewed_at, score, status,
                        mastery_before, mastery_after,
                        stability_before, stability_after,
                        sentence, reference_translation, user_translation, feedback
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_review_key,
                        card_id,
                        _safe_text(snapshot_word),
                        _safe_text(snapshot_pos),
                        _safe_text(snapshot_meaning),
                        format_utc(after.last_reviewed_at),
                        normalized_score,
                        status,
                        before_mastery,
                        after.mastery,
                        before_stability,
                        after.stability_days,
                        sentence,
                        reference_translation,
                        user_translation,
                        feedback,
                    ),
                )
                return after

        return self._run_with_recovery(operation)

    def list_review_events(
        self,
        *,
        card_id: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[ReviewEvent]:
        if limit is not None and limit <= 0:
            return []

        def operation() -> list[ReviewEvent]:
            where = " WHERE card_id = ?" if card_id is not None else ""
            order = "DESC" if newest_first else "ASC"
            sql = f"SELECT * FROM review_events{where} ORDER BY event_id {order}"
            parameters: list[Any] = [] if card_id is None else [str(card_id)]
            if limit is not None:
                sql += " LIMIT ?"
                parameters.append(int(limit))
            with closing(self._connect()) as connection:
                rows = connection.execute(sql, parameters).fetchall()
            return [self._row_to_event(row) for row in rows]

        return self._run_with_recovery(operation)

    def get_review_events(self, **kwargs: Any) -> list[ReviewEvent]:
        return self.list_review_events(**kwargs)

    def reset_progress(self) -> int:
        """Clear scheduling state while preserving immutable review history."""

        def operation() -> int:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute("DELETE FROM progress")
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('progress_reset_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (format_utc(utc_now()),),
                )
                return max(0, cursor.rowcount)

        return self._run_with_recovery(operation)

    def clear_history(self) -> int:
        """Clear review events while retaining current scheduling state."""

        def operation() -> int:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute("DELETE FROM review_events")
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('history_cleared_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (format_utc(utc_now()),),
                )
                return max(0, cursor.rowcount)

        return self._run_with_recovery(operation)

    def clear_all(self) -> dict[str, int]:
        """Clear all user learning data but retain schema metadata."""

        def operation() -> dict[str, int]:
            with closing(self._connect()) as connection, connection:
                progress_cursor = connection.execute("DELETE FROM progress")
                history_cursor = connection.execute("DELETE FROM review_events")
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('all_cleared_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (format_utc(utc_now()),),
                )
                return {
                    "progress": max(0, progress_cursor.rowcount),
                    "review_events": max(0, history_cursor.rowcount),
                }

        return self._run_with_recovery(operation)
