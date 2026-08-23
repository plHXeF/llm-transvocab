"""SQLite persistence for current learning state and immutable review history."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Iterable, TypeVar
import warnings

from domain import (
    DEFAULT_DIFFICULTY,
    DEFAULT_STABILITY_DAYS,
    MIN_DIFFICULTY_SAMPLES,
    Card,
    Progress,
    ReviewEvent,
    ensure_utc,
    evolve_progress,
    format_utc,
    parse_utc,
    utc_now,
)


SCHEMA_VERSION = 4
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "learning.db"
VALID_REVIEW_STATUSES = frozenset({"answered", "skipped"})
DIFFICULTY_MODEL_VERSION = "personal-difficulty.v1"
DIFFICULTY_MIN_GLOBAL_SAMPLES = 30
DIFFICULTY_RETRAIN_INTERVAL = 20
DIFFICULTY_MAX_TRAINING_EVENTS = 1_000
MIN_ATTRIBUTION_CONFIDENCE = 0.25
DEFAULT_PERSONAL_BASELINE = 0.70

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


def _optional_unit_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = _safe_float(value, math.nan)
    if not math.isfinite(converted):
        return None
    return min(1.0, max(0.0, converted))


def _safe_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item)[:40] for item in decoded[:5] if str(item).strip())


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
                    updated_at TEXT NOT NULL,
                    difficulty REAL NOT NULL DEFAULT 0.5,
                    difficulty_samples INTEGER NOT NULL DEFAULT 0
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
                    feedback TEXT,
                    target_error_weight REAL,
                    attribution_confidence REAL,
                    non_target_error_tags TEXT,
                    target_performance REAL,
                    expected_performance REAL,
                    effective_mastery_before REAL,
                    attempts_before INTEGER NOT NULL DEFAULT 0,
                    evaluator_profile TEXT,
                    meaning_revealed INTEGER NOT NULL DEFAULT 0
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
                # A missing meta row may belong to a partially initialized old
                # database. Column-aware migrations are idempotent for a truly
                # new database and repair that case without discarding data.
                self._migrate(connection, 0)
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

        if version < 3:
            progress_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(progress)").fetchall()
            }
            if "difficulty" not in progress_columns:
                connection.execute(
                    "ALTER TABLE progress ADD COLUMN difficulty REAL DEFAULT 0.5"
                )
            if "difficulty_samples" not in progress_columns:
                connection.execute(
                    "ALTER TABLE progress ADD COLUMN difficulty_samples "
                    "INTEGER DEFAULT 0"
                )
            # Some SQLite versions report a failed quick_check when a
            # NOT NULL column is added directly to a populated legacy table.
            # Explicitly backfill migrated rows; newly created databases keep
            # the stricter declarations from CREATE TABLE above.
            connection.execute(
                "UPDATE progress SET difficulty = 0.5 WHERE difficulty IS NULL"
            )
            connection.execute(
                "UPDATE progress SET difficulty_samples = 0 "
                "WHERE difficulty_samples IS NULL"
            )

            event_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(review_events)"
                ).fetchall()
            }
            additions = {
                "target_error_weight": "REAL",
                "attribution_confidence": "REAL",
                "non_target_error_tags": "TEXT",
                "target_performance": "REAL",
                "expected_performance": "REAL",
                "effective_mastery_before": "REAL",
                "attempts_before": "INTEGER DEFAULT 0",
                "evaluator_profile": "TEXT",
            }
            for column, declaration in additions.items():
                if column not in event_columns:
                    connection.execute(
                        f"ALTER TABLE review_events ADD COLUMN {column} {declaration}"
                    )
            connection.execute(
                "UPDATE review_events SET attempts_before = 0 "
                "WHERE attempts_before IS NULL"
            )
            version = 3

        if version < 4:
            event_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(review_events)"
                ).fetchall()
            }
            if "meaning_revealed" not in event_columns:
                connection.execute(
                    "ALTER TABLE review_events ADD COLUMN meaning_revealed "
                    "INTEGER DEFAULT 0"
                )
            connection.execute(
                "UPDATE review_events SET meaning_revealed = 0 "
                "WHERE meaning_revealed IS NULL"
            )
            version = 4

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
            difficulty=row["difficulty"],
            difficulty_samples=row["difficulty_samples"],
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
            target_error_weight=_optional_unit_float(row["target_error_weight"]),
            attribution_confidence=_optional_unit_float(
                row["attribution_confidence"]
            ),
            non_target_error_tags=_safe_tags(row["non_target_error_tags"]),
            target_performance=_optional_unit_float(row["target_performance"]),
            expected_performance=_optional_unit_float(row["expected_performance"]),
            effective_mastery_before=_optional_unit_float(
                row["effective_mastery_before"]
            ),
            attempts_before=max(0, int(_safe_float(row["attempts_before"], 0.0))),
            evaluator_profile=(
                row["evaluator_profile"]
                if isinstance(row["evaluator_profile"], str)
                else None
            ),
            meaning_revealed=bool(row["meaning_revealed"]),
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

    @staticmethod
    def _difficulty_epoch(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'difficulty_epoch_event_id'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return max(0, int(row["value"]))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _attempt_feature(attempts: int) -> float:
        return min(1.0, math.log1p(max(0, attempts)) / math.log(11.0))

    @staticmethod
    def _effective_mastery_before(
        progress: Progress | None,
        reviewed_at: datetime,
    ) -> float:
        if (
            progress is None
            or progress.attempts <= 0
            or progress.last_reviewed_at is None
        ):
            return 0.0
        elapsed_days = max(
            0.0,
            (
                ensure_utc(reviewed_at) - ensure_utc(progress.last_reviewed_at)
            ).total_seconds()
            / 86_400.0,
        )
        stability = max(0.25, float(progress.stability_days))
        retention = math.exp(-elapsed_days / stability)
        return min(1.0, max(0.0, progress.mastery * retention))

    @classmethod
    def _load_difficulty_model(
        cls,
        connection: sqlite3.Connection,
    ) -> dict[str, float | int | str]:
        epoch = cls._difficulty_epoch(connection)
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'difficulty_model'"
        ).fetchone()
        if row is not None:
            try:
                candidate = json.loads(row["value"])
                if (
                    isinstance(candidate, dict)
                    and candidate.get("version") == DIFFICULTY_MODEL_VERSION
                    and int(candidate.get("epoch_event_id", -1)) == epoch
                ):
                    return {
                        "version": DIFFICULTY_MODEL_VERSION,
                        "epoch_event_id": epoch,
                        "sample_count": max(0, int(candidate.get("sample_count", 0))),
                        "baseline": min(
                            0.98,
                            max(0.05, float(candidate.get("baseline", DEFAULT_PERSONAL_BASELINE))),
                        ),
                        "mean_effective_mastery": min(
                            1.0,
                            max(0.0, float(candidate.get("mean_effective_mastery", 0.0))),
                        ),
                        "mean_attempt_feature": min(
                            1.0,
                            max(0.0, float(candidate.get("mean_attempt_feature", 0.0))),
                        ),
                        "effective_mastery_coefficient": min(
                            1.0,
                            max(
                                0.0,
                                float(candidate.get("effective_mastery_coefficient", 0.35)),
                            ),
                        ),
                        "attempt_coefficient": min(
                            0.30,
                            max(0.0, float(candidate.get("attempt_coefficient", 0.05))),
                        ),
                    }
            except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
                pass

        rows = connection.execute(
            """
            SELECT target_performance, attribution_confidence,
                   effective_mastery_before, attempts_before
            FROM review_events
            WHERE event_id > ? AND status = 'answered'
              AND target_performance IS NOT NULL
              AND attribution_confidence >= ?
              AND meaning_revealed = 0
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (epoch, MIN_ATTRIBUTION_CONFIDENCE, DIFFICULTY_MAX_TRAINING_EVENTS),
        ).fetchall()
        if not rows:
            return {
                "version": DIFFICULTY_MODEL_VERSION,
                "epoch_event_id": epoch,
                "sample_count": 0,
                "baseline": DEFAULT_PERSONAL_BASELINE,
                "mean_effective_mastery": 0.0,
                "mean_attempt_feature": 0.0,
                "effective_mastery_coefficient": 0.35,
                "attempt_coefficient": 0.05,
            }
        weights = [
            max(MIN_ATTRIBUTION_CONFIDENCE, _safe_float(row["attribution_confidence"]))
            for row in rows
        ]
        total_weight = sum(weights)
        baseline_prior_weight = 10.0
        return {
            "version": DIFFICULTY_MODEL_VERSION,
            "epoch_event_id": epoch,
            "sample_count": len(rows),
            "baseline": (
                baseline_prior_weight * DEFAULT_PERSONAL_BASELINE
                + sum(
                    weight
                    * _safe_float(
                        row["target_performance"], DEFAULT_PERSONAL_BASELINE
                    )
                    for row, weight in zip(rows, weights)
                )
            )
            / (baseline_prior_weight + total_weight),
            "mean_effective_mastery": sum(
                weight * _safe_float(row["effective_mastery_before"])
                for row, weight in zip(rows, weights)
            )
            / total_weight,
            "mean_attempt_feature": sum(
                weight * cls._attempt_feature(int(_safe_float(row["attempts_before"])))
                for row, weight in zip(rows, weights)
            )
            / total_weight,
            "effective_mastery_coefficient": 0.35,
            "attempt_coefficient": 0.05,
        }

    @classmethod
    def _expected_performance(
        cls,
        model: dict[str, float | int | str],
        effective_mastery: float,
        attempts_before: int,
    ) -> float:
        expected = (
            float(model["baseline"])
            + float(model["effective_mastery_coefficient"])
            * (effective_mastery - float(model["mean_effective_mastery"]))
            + float(model["attempt_coefficient"])
            * (
                cls._attempt_feature(attempts_before)
                - float(model["mean_attempt_feature"])
            )
        )
        return min(0.98, max(0.05, expected))

    @classmethod
    def _refresh_card_difficulty(
        cls,
        connection: sqlite3.Connection,
        card_id: str,
    ) -> None:
        epoch = cls._difficulty_epoch(connection)
        rows = connection.execute(
            """
            SELECT card_id, score, target_error_weight, attribution_confidence,
                   target_performance, expected_performance, meaning_revealed
            FROM review_events
            WHERE event_id > ? AND status = 'answered'
              AND target_performance IS NOT NULL
              AND expected_performance IS NOT NULL
              AND attribution_confidence >= ?
            """,
            (epoch, MIN_ATTRIBUTION_CONFIDENCE),
        ).fetchall()
        card_rows = [row for row in rows if row["card_id"] == card_id]
        samples = len(card_rows)
        difficulty = DEFAULT_DIFFICULTY
        if samples >= MIN_DIFFICULTY_SAMPLES:
            residual_weight = sum(
                max(MIN_ATTRIBUTION_CONFIDENCE, _safe_float(row["attribution_confidence"]))
                for row in card_rows
            )
            residual = sum(
                max(MIN_ATTRIBUTION_CONFIDENCE, _safe_float(row["attribution_confidence"]))
                * (
                    _safe_float(row["expected_performance"], DEFAULT_PERSONAL_BASELINE)
                    - (
                        0.0
                        if bool(row["meaning_revealed"])
                        else _safe_float(
                            row["target_performance"], DEFAULT_PERSONAL_BASELINE
                        )
                    )
                )
                for row in card_rows
            ) / max(residual_weight, 1e-9)

            def attributed_error_average(source: list[sqlite3.Row]) -> float | None:
                weighted_sum = 0.0
                total = 0.0
                for row in source:
                    sentence_loss = (
                        1.0
                        if bool(row["meaning_revealed"])
                        else max(0.0, 1.0 - _safe_float(row["score"]) / 100.0)
                    )
                    weight = (
                        sentence_loss
                        * max(
                            MIN_ATTRIBUTION_CONFIDENCE,
                            _safe_float(row["attribution_confidence"]),
                        )
                    )
                    weighted_sum += weight * _safe_float(row["target_error_weight"])
                    total += weight
                return None if total <= 1e-9 else weighted_sum / total

            card_attribution = attributed_error_average(card_rows)
            global_attribution = attributed_error_average(rows)
            attribution_excess = (
                0.0
                if card_attribution is None or global_attribution is None
                else card_attribution - global_attribution
            )
            reliability = samples / (samples + 5.0)
            signal = 0.80 * residual + 0.20 * attribution_excess
            difficulty = min(1.0, max(0.0, 0.50 + reliability * signal))

        connection.execute(
            "UPDATE progress SET difficulty = ?, difficulty_samples = ? WHERE card_id = ?",
            (difficulty, samples, card_id),
        )

    @classmethod
    def _difficulty_event_count(cls, connection: sqlite3.Connection) -> int:
        epoch = cls._difficulty_epoch(connection)
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM review_events
            WHERE event_id > ? AND status = 'answered'
              AND target_performance IS NOT NULL
              AND attribution_confidence >= ?
              AND meaning_revealed = 0
            """,
            (epoch, MIN_ATTRIBUTION_CONFIDENCE),
        ).fetchone()
        return 0 if row is None else max(0, int(row["count"]))

    @classmethod
    def _difficulty_recalibration_due_in_connection(
        cls,
        connection: sqlite3.Connection,
    ) -> bool:
        count = cls._difficulty_event_count(connection)
        if count < DIFFICULTY_MIN_GLOBAL_SAMPLES:
            return False
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'difficulty_model'"
        ).fetchone()
        if row is None:
            return True
        try:
            model = json.loads(row["value"])
            if (
                not isinstance(model, dict)
                or model.get("version") != DIFFICULTY_MODEL_VERSION
                or int(model.get("epoch_event_id", -1))
                != cls._difficulty_epoch(connection)
            ):
                return True
            trained_count = int(model.get("sample_count", 0))
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
            return True
        return trained_count > count or count - trained_count >= DIFFICULTY_RETRAIN_INTERVAL

    def difficulty_recalibration_due(self) -> bool:
        def operation() -> bool:
            with closing(self._connect()) as connection:
                return self._difficulty_recalibration_due_in_connection(connection)

        return self._run_with_recovery(operation)

    def difficulty_status(self) -> dict[str, Any]:
        """Return safe, user-facing calibration progress without event contents."""

        def operation() -> dict[str, Any]:
            with closing(self._connect()) as connection:
                count = self._difficulty_event_count(connection)
                row = connection.execute(
                    "SELECT value FROM meta WHERE key = 'difficulty_model'"
                ).fetchone()
                trained = False
                trained_count = 0
                baseline: float | None = None
                if row is not None:
                    try:
                        model = json.loads(row["value"])
                        trained = bool(
                            isinstance(model, dict)
                            and model.get("version") == DIFFICULTY_MODEL_VERSION
                            and int(model.get("epoch_event_id", -1))
                            == self._difficulty_epoch(connection)
                        )
                        if trained:
                            trained_count = max(0, int(model.get("sample_count", 0)))
                            baseline = min(
                                1.0,
                                max(0.0, float(model.get("baseline"))),
                            )
                    except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
                        trained = False
                next_training_at = (
                    DIFFICULTY_MIN_GLOBAL_SAMPLES
                    if not trained
                    else max(
                        DIFFICULTY_MIN_GLOBAL_SAMPLES,
                        trained_count + DIFFICULTY_RETRAIN_INTERVAL,
                    )
                )
                return {
                    "valid_samples": count,
                    "trained": trained,
                    "trained_samples": trained_count,
                    "baseline": baseline,
                    "next_training_at": next_training_at,
                }

        return self._run_with_recovery(operation)

    def recalibrate_difficulty(self, *, force: bool = False) -> bool:
        """Fit the tiny personal calibration model and refresh item difficulty."""

        def operation() -> bool:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                if not force and not self._difficulty_recalibration_due_in_connection(
                    connection
                ):
                    return False
                epoch = self._difficulty_epoch(connection)
                rows = connection.execute(
                    """
                    SELECT event_id, target_performance, attribution_confidence,
                           effective_mastery_before, attempts_before
                    FROM review_events
                    WHERE event_id > ? AND status = 'answered'
                      AND target_performance IS NOT NULL
                      AND attribution_confidence >= ?
                      AND meaning_revealed = 0
                    ORDER BY event_id DESC
                    LIMIT ?
                    """,
                    (
                        epoch,
                        MIN_ATTRIBUTION_CONFIDENCE,
                        DIFFICULTY_MAX_TRAINING_EVENTS,
                    ),
                ).fetchall()
                if not rows:
                    return False

                weighted_rows: list[tuple[sqlite3.Row, float]] = []
                for age, row in enumerate(rows):
                    confidence = max(
                        MIN_ATTRIBUTION_CONFIDENCE,
                        _safe_float(row["attribution_confidence"]),
                    )
                    weighted_rows.append((row, confidence * (0.995**age)))
                total_weight = sum(weight for _, weight in weighted_rows)
                baseline = sum(
                    weight * _safe_float(row["target_performance"], DEFAULT_PERSONAL_BASELINE)
                    for row, weight in weighted_rows
                ) / total_weight
                mean_effective = sum(
                    weight * _safe_float(row["effective_mastery_before"])
                    for row, weight in weighted_rows
                ) / total_weight
                mean_attempt = sum(
                    weight
                    * self._attempt_feature(int(_safe_float(row["attempts_before"])))
                    for row, weight in weighted_rows
                ) / total_weight

                s11 = s12 = s22 = t1 = t2 = 0.0
                for row, weight in weighted_rows:
                    x1 = _safe_float(row["effective_mastery_before"]) - mean_effective
                    x2 = (
                        self._attempt_feature(int(_safe_float(row["attempts_before"])))
                        - mean_attempt
                    )
                    centered_y = _safe_float(row["target_performance"]) - baseline
                    s11 += weight * x1 * x1
                    s12 += weight * x1 * x2
                    s22 += weight * x2 * x2
                    t1 += weight * x1 * centered_y
                    t2 += weight * x2 * centered_y
                ridge = max(1.0, total_weight * 0.15)
                s11 += ridge
                s22 += ridge
                determinant = s11 * s22 - s12 * s12
                if abs(determinant) <= 1e-12:
                    effective_coefficient, attempt_coefficient = 0.35, 0.05
                else:
                    effective_coefficient = (t1 * s22 - t2 * s12) / determinant
                    attempt_coefficient = (s11 * t2 - s12 * t1) / determinant
                effective_coefficient = min(1.0, max(0.0, effective_coefficient))
                attempt_coefficient = min(0.30, max(0.0, attempt_coefficient))
                total_count = self._difficulty_event_count(connection)
                model: dict[str, float | int | str] = {
                    "version": DIFFICULTY_MODEL_VERSION,
                    "epoch_event_id": epoch,
                    "sample_count": total_count,
                    "baseline": min(0.98, max(0.05, baseline)),
                    "mean_effective_mastery": min(1.0, max(0.0, mean_effective)),
                    "mean_attempt_feature": min(1.0, max(0.0, mean_attempt)),
                    "effective_mastery_coefficient": effective_coefficient,
                    "attempt_coefficient": attempt_coefficient,
                }
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('difficulty_model', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (json.dumps(model, ensure_ascii=False, separators=(",", ":")),),
                )
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('difficulty_model_updated_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (format_utc(utc_now()),),
                )

                all_rows = connection.execute(
                    """
                    SELECT event_id, effective_mastery_before, attempts_before
                    FROM review_events
                    WHERE event_id > ? AND status = 'answered'
                      AND target_performance IS NOT NULL
                      AND attribution_confidence >= ?
                      AND meaning_revealed = 0
                    """,
                    (epoch, MIN_ATTRIBUTION_CONFIDENCE),
                ).fetchall()
                for row in all_rows:
                    expected = self._expected_performance(
                        model,
                        _safe_float(row["effective_mastery_before"]),
                        int(_safe_float(row["attempts_before"])),
                    )
                    connection.execute(
                        "UPDATE review_events SET expected_performance = ? WHERE event_id = ?",
                        (expected, row["event_id"]),
                    )
                progress_rows = connection.execute(
                    "SELECT card_id FROM progress"
                ).fetchall()
                for row in progress_rows:
                    self._refresh_card_difficulty(connection, row["card_id"])
                return True

        return self._run_with_recovery(operation)

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
        target_error_weight: float | None = None,
        attribution_confidence: float | None = None,
        non_target_error_tags: Iterable[str] = (),
        evaluator_profile: str | None = None,
        meaning_revealed: bool = False,
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
        normalized_error_weight: float | None = None
        normalized_confidence: float | None = None
        normalized_tags: tuple[str, ...] = ()
        normalized_profile: str | None = None
        normalized_meaning_revealed = bool(
            meaning_revealed and status == "answered"
        )
        if status == "answered" and (
            target_error_weight is not None or attribution_confidence is not None
        ):
            if target_error_weight is None or attribution_confidence is None:
                raise ValueError(
                    "target_error_weight and attribution_confidence must be provided together"
                )
            normalized_error_weight = _optional_unit_float(target_error_weight)
            normalized_confidence = _optional_unit_float(attribution_confidence)
            if normalized_error_weight is None or normalized_confidence is None:
                raise ValueError("attribution values must be finite numbers between 0 and 1")
            allowed_tags = {
                "context_vocabulary",
                "grammar",
                "omission",
                "chinese_expression",
                "overtranslation",
                "other",
            }
            collected_tags: list[str] = []
            for item in non_target_error_tags:
                tag = str(item).strip()
                if tag not in allowed_tags:
                    raise ValueError(f"unsupported non-target error tag: {tag}")
                if tag not in collected_tags:
                    collected_tags.append(tag)
                if len(collected_tags) >= 5:
                    break
            normalized_tags = tuple(collected_tags)
            normalized_profile = _safe_text(evaluator_profile).strip()[:200] or None
        timestamp = ensure_utc(reviewed_at or utc_now())

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
                attempts_before = 0 if before is None else before.attempts
                effective_mastery_before = self._effective_mastery_before(
                    before,
                    timestamp,
                )
                target_performance: float | None = None
                expected_performance: float | None = None
                if (
                    normalized_error_weight is not None
                    and normalized_confidence is not None
                ):
                    sentence_loss = 1.0 - normalized_score / 100.0
                    target_performance = min(
                        1.0,
                        max(0.0, 1.0 - sentence_loss * normalized_error_weight),
                    )
                    model = self._load_difficulty_model(connection)
                    expected_performance = self._expected_performance(
                        model,
                        effective_mastery_before,
                        attempts_before,
                    )
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
                        last_reviewed_at, last_score, updated_at,
                        difficulty, difficulty_samples
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(card_id) DO UPDATE SET
                        attempts = excluded.attempts,
                        mastery = excluded.mastery,
                        stability_days = excluded.stability_days,
                        last_reviewed_at = excluded.last_reviewed_at,
                        last_score = excluded.last_score,
                        updated_at = excluded.updated_at,
                        difficulty = excluded.difficulty,
                        difficulty_samples = excluded.difficulty_samples
                    """,
                    (
                        after.card_id,
                        after.attempts,
                        after.mastery,
                        after.stability_days,
                        format_utc(after.last_reviewed_at),
                        after.last_score,
                        format_utc(after.updated_at),
                        after.difficulty,
                        after.difficulty_samples,
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
                        sentence, reference_translation, user_translation, feedback,
                        target_error_weight, attribution_confidence,
                        non_target_error_tags, target_performance,
                        expected_performance, effective_mastery_before,
                        attempts_before, evaluator_profile, meaning_revealed
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        normalized_error_weight,
                        normalized_confidence,
                        (
                            None
                            if normalized_error_weight is None
                            else json.dumps(
                                normalized_tags,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        ),
                        target_performance,
                        expected_performance,
                        (
                            None
                            if normalized_error_weight is None
                            else effective_mastery_before
                        ),
                        attempts_before,
                        normalized_profile,
                        int(normalized_meaning_revealed),
                    ),
                )
                self._refresh_card_difficulty(connection, card_id)
                refreshed = connection.execute(
                    "SELECT * FROM progress WHERE card_id = ?", (card_id,)
                ).fetchone()
                return after if refreshed is None else self._row_to_progress(refreshed)

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
        """Clear scheduling/difficulty state while preserving visible history."""

        def operation() -> int:
            with closing(self._connect()) as connection, connection:
                latest = connection.execute(
                    "SELECT COALESCE(MAX(event_id), 0) AS event_id FROM review_events"
                ).fetchone()
                epoch_event_id = 0 if latest is None else int(latest["event_id"])
                cursor = connection.execute("DELETE FROM progress")
                connection.execute(
                    "DELETE FROM meta WHERE key IN "
                    "('difficulty_model', 'difficulty_model_updated_at')"
                )
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('difficulty_epoch_event_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(epoch_event_id),),
                )
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
                    "DELETE FROM meta WHERE key IN "
                    "('difficulty_model', 'difficulty_model_updated_at')"
                )
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('difficulty_epoch_event_id', '0') "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                )
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
