"""Redacted, bounded diagnostics for model-facing failures.

The log intentionally stores no prompts, completions, translations, request
headers, or API keys.  It is a local troubleshooting aid, not an audit trail.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from llm_service import LLMSettings, redact_text


LOG_SCHEMA_VERSION = 2
DEFAULT_MAX_ENTRIES = 500
_WRITE_LOCK = threading.RLock()
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|token|key)(\s*[=:]\s*)[^\s&;,]+"
)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_endpoint(value: str | None) -> str:
    """Keep only the endpoint origin; discard credentials, query, and path."""

    if not value:
        return ""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if not host:
            return ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme.lower()}://{host}{port}"
    except (TypeError, ValueError):
        return ""


def _redact_for_log(value: object, secrets: Iterable[str] = ()) -> str:
    safe = str(value)
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[REDACTED]")
    safe = redact_text(safe)
    safe = _BEARER_PATTERN.sub(r"\1[REDACTED]", safe)
    safe = _QUERY_SECRET_PATTERN.sub(r"\1\2[REDACTED]", safe)
    return safe[:2_000]


def _optional_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _optional_int(value: object, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= minimum else None


def _status_code(error: object) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _request_id(error: object) -> str | None:
    direct = getattr(error, "request_id", None)
    if direct:
        return _optional_text(direct, limit=200)
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            return _optional_text(
                headers.get("x-request-id") or headers.get("request-id"),
                limit=200,
            )
        except (AttributeError, TypeError):
            pass
    return None


def _category(error: object, status_code: int | None) -> str:
    explicit = _optional_text(getattr(error, "category", None), limit=80)
    if explicit:
        return explicit
    name = type(error).__name__.casefold()
    if "validation" in name or isinstance(error, (ValueError, TypeError)):
        return "response_validation"
    if "timeout" in name:
        return "timeout"
    if "ratelimit" in name or status_code == 429:
        return "rate_limit"
    if "authentication" in name or status_code in {401, 403}:
        return "authentication"
    if "connection" in name:
        return "connection"
    if status_code is not None:
        return "api_status"
    return "unknown"


@dataclass(frozen=True)
class ModelErrorEntry:
    timestamp: str
    operation: str
    category: str
    error_type: str
    message: str
    endpoint: str = ""
    model: str = ""
    reasoning_effort: str = ""
    prompt_version: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: int | None = None
    attempts: int | None = None
    card_id: str | None = None
    batch_id: str | None = None
    schema_version: int = LOG_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, raw: Any) -> "ModelErrorEntry | None":
        if not isinstance(raw, dict):
            return None
        required = ("timestamp", "operation", "category", "error_type", "message")
        if any(not isinstance(raw.get(key), str) or not raw[key] for key in required):
            return None
        status = _optional_int(raw.get("status_code"), minimum=100)
        return cls(
            timestamp=raw["timestamp"][:80],
            operation=raw["operation"][:120],
            category=raw["category"][:80],
            error_type=raw["error_type"][:160],
            message=raw["message"][:2_000],
            endpoint=str(raw.get("endpoint") or "")[:500],
            model=str(raw.get("model") or "")[:300],
            reasoning_effort=str(raw.get("reasoning_effort") or "")[:80],
            prompt_version=_optional_text(raw.get("prompt_version"), limit=120),
            status_code=status,
            request_id=_optional_text(raw.get("request_id"), limit=200),
            finish_reason=_optional_text(raw.get("finish_reason"), limit=120),
            completion_tokens=_optional_int(raw.get("completion_tokens")),
            reasoning_tokens=_optional_int(raw.get("reasoning_tokens")),
            latency_ms=_optional_int(raw.get("latency_ms")),
            attempts=_optional_int(raw.get("attempts"), minimum=1),
            card_id=_optional_text(raw.get("card_id"), limit=160),
            batch_id=_optional_text(raw.get("batch_id"), limit=160),
            schema_version=LOG_SCHEMA_VERSION,
        )


class ModelErrorLog:
    """A small JSONL store with atomic writes and tolerant reads."""

    def __init__(self, path: str | Path, *, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.path = Path(path).expanduser()
        self.max_entries = max(1, int(max_entries))

    def record(
        self,
        operation: str,
        error: object,
        settings: LLMSettings | None = None,
        *,
        prompt_version: str | None = None,
        card_id: str | None = None,
        batch_id: str | None = None,
        category: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        error_type: str | None = None,
    ) -> ModelErrorEntry:
        settings = settings or LLMSettings()
        detected_status = status_code if status_code is not None else _status_code(error)
        safe_request_id = (
            _optional_text(request_id, limit=200)
            if request_id is not None
            else _request_id(error)
        )
        entry = ModelErrorEntry(
            timestamp=_utc_now_text(),
            operation=str(operation).strip()[:120] or "unknown",
            category=(category or _category(error, detected_status))[:80],
            error_type=(error_type or type(error).__name__)[:160],
            message=_redact_for_log(error, secrets=(settings.api_key,)),
            endpoint=_safe_endpoint(settings.resolved_base_url),
            model=_redact_for_log(
                settings.model, secrets=(settings.api_key,)
            )[:300],
            reasoning_effort=settings.reasoning_effort.value,
            prompt_version=_optional_text(prompt_version, limit=120),
            status_code=detected_status,
            request_id=(
                None
                if safe_request_id is None
                else _redact_for_log(safe_request_id, secrets=(settings.api_key,))
            ),
            finish_reason=_optional_text(
                getattr(error, "finish_reason", None), limit=120
            ),
            completion_tokens=_optional_int(
                getattr(error, "completion_tokens", None)
            ),
            reasoning_tokens=_optional_int(
                getattr(error, "reasoning_tokens", None)
            ),
            latency_ms=_optional_int(getattr(error, "latency_ms", None)),
            attempts=_optional_int(getattr(error, "attempts", None), minimum=1),
            card_id=_optional_text(card_id, limit=160),
            batch_id=_optional_text(batch_id, limit=160),
        )
        with _WRITE_LOCK:
            entries = self.list_entries()
            entries.append(entry)
            self._write(entries[-self.max_entries :])
        return entry

    def list_entries(self, *, limit: int | None = None) -> list[ModelErrorEntry]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError):
            return []
        entries: list[ModelErrorEntry] = []
        for line in lines:
            try:
                entry = ModelErrorEntry.from_mapping(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if entry is not None:
                entries.append(entry)
        if limit is not None:
            return entries[-max(0, int(limit)) :]
        return entries

    def export_jsonl(self) -> str:
        return "".join(
            json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n"
            for entry in self.list_entries()
        )

    def clear(self) -> int:
        with _WRITE_LOCK:
            count = len(self.list_entries())
            self._write(())
        return count

    def _write(self, entries: Iterable[ModelErrorEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n"
            for entry in entries
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


__all__ = ["ModelErrorEntry", "ModelErrorLog"]
