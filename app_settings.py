"""Local application settings with split, atomic, owner-only persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import config
from llm_service import (
    LLMConfigurationError,
    LLMSettings,
    SentenceDifficulty,
    coerce_sentence_difficulty,
    mask_api_key,
    redact_text,
)


SETTINGS_FORMAT_VERSION = 3
LEGACY_SETTINGS_FORMAT_VERSION = 1
PREVIOUS_SETTINGS_FORMAT_VERSION = 2
KEYS_FORMAT_VERSION = 1
KeyAction = Literal["replace", "keep", "delete"]
ApiKeySource = Literal["local", "environment", "none"]


class SettingsError(RuntimeError):
    """A local settings file could not be safely loaded or saved."""


@dataclass(frozen=True)
class AppSettings:
    llm: LLMSettings
    sentence_difficulty: SentenceDifficulty | str = SentenceDifficulty.CET6_POSTGRAD
    version: int = SETTINGS_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sentence_difficulty",
            coerce_sentence_difficulty(self.sentence_difficulty),
        )

    def safe_dict(self) -> dict[str, Any]:
        """Return a log/UI-safe representation with a masked API key."""

        return {
            "version": self.version,
            "sentence_difficulty": self.sentence_difficulty.value,
            "llm": {
                "model": self.llm.model,
                "api_key": mask_api_key(self.llm.api_key),
                "base_url": self.llm.base_url,
                "reasoning_effort": self.llm.reasoning_effort.value,
                "timeout_seconds": self.llm.timeout_seconds,
                "max_retries": self.llm.max_retries,
            },
        }

    def _settings_storage_dict(self) -> dict[str, Any]:
        """Serialize non-secret settings; this payload must never contain a key."""

        return {
            "version": self.version,
            "sentence_difficulty": self.sentence_difficulty.value,
            "llm": {
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "reasoning_effort": self.llm.reasoning_effort.value,
                "timeout_seconds": self.llm.timeout_seconds,
                "max_retries": self.llm.max_retries,
            },
        }


class SettingsStore:
    """Persist ordinary settings and endpoint-scoped API keys separately."""

    def __init__(
        self,
        settings_path: str | os.PathLike[str] | None = None,
        keys_path: str | os.PathLike[str] | None = None,
    ):
        self.settings_path = Path(
            settings_path or config.APP_SETTINGS_FILE
        ).expanduser()
        self.keys_path = Path(keys_path or config.API_KEYS_FILE).expanduser()
        self.legacy_keys_path = (
            None
            if keys_path is not None
            else Path(config.LEGACY_PROVIDER_KEYS_FILE).expanduser()
        )
        if self.settings_path == self.keys_path:
            raise SettingsError("普通设置文件与 API key 文件不能使用同一路径")

    @property
    def path(self) -> Path:
        """Backward-friendly alias for the ordinary settings path."""

        return self.settings_path

    def load(self) -> AppSettings | None:
        if not self.settings_path.exists():
            return None
        try:
            settings_payload = _read_owner_only_json(self.settings_path)
            base_url = _endpoint_from_storage(settings_payload)
            api_key = self.load_api_key(base_url)
            return _settings_from_storage(settings_payload, api_key=api_key)
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            LLMConfigurationError,
        ) as exc:
            raise SettingsError(
                f"无法读取本地设置：{redact_text(exc)}"
            ) from None

    def load_or_default(self) -> AppSettings:
        loaded = self.load()
        if loaded is not None:
            return loaded
        defaults = default_app_settings()
        api_key = self.load_api_key(defaults.llm.resolved_base_url)
        return AppSettings(
            version=defaults.version,
            llm=LLMSettings(
                base_url=defaults.llm.base_url,
                model=defaults.llm.model,
                api_key=api_key,
                reasoning_effort=defaults.llm.reasoning_effort,
                timeout_seconds=defaults.llm.timeout_seconds,
                max_retries=defaults.llm.max_retries,
            ),
        )

    def load_api_key(
        self,
        base_url: str | None,
    ) -> str:
        """Load one endpoint's saved key, falling back to its environment key."""

        try:
            key_map = self._load_key_map()
            for slot in _candidate_key_slots(base_url):
                if slot in key_map:
                    return key_map[slot]
            return _environment_key(base_url)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SettingsError(
                f"无法读取 API key：{redact_text(exc)}"
            ) from None

    def api_key_source(
        self,
        base_url: str | None,
    ) -> ApiKeySource:
        """Report where the effective key comes from without returning it."""

        try:
            key_map = self._load_key_map()
            if any(slot in key_map for slot in _candidate_key_slots(base_url)):
                return "local"
            if _environment_key(base_url):
                return "environment"
            return "none"
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SettingsError(
                f"无法读取 API key 来源：{redact_text(exc)}"
            ) from None

    def save(
        self,
        settings: AppSettings,
        *,
        key_action: KeyAction = "replace",
    ) -> None:
        """Save ordinary settings and explicitly manage the endpoint key.

        ``replace`` keeps the historical behavior: store ``llm.api_key`` or
        delete the slot when it is empty. ``keep`` never reads or writes the
        key document. ``delete`` removes only the selected endpoint URL
        slot, allowing the environment fallback to become effective.
        """

        if settings.version != SETTINGS_FORMAT_VERSION:
            raise SettingsError(
                f"不支持的设置版本 {settings.version}；当前版本为 {SETTINGS_FORMAT_VERSION}"
            )
        if key_action not in {"replace", "keep", "delete"}:
            raise SettingsError(
                "key_action 必须是 'replace'、'keep' 或 'delete'"
            )
        try:
            settings_payload = settings._settings_storage_dict()

            if key_action != "keep":
                key_map = self._load_key_map()
                slot = _key_slot(settings.llm.resolved_base_url)
                if key_action == "replace" and settings.llm.api_key:
                    key_map[slot] = settings.llm.api_key
                else:
                    for candidate in _candidate_key_slots(
                        settings.llm.resolved_base_url
                    ):
                        key_map.pop(candidate, None)
                key_payload = {
                    "version": KEYS_FORMAT_VERSION,
                    "keys": dict(sorted(key_map.items())),
                }

                # Write the endpoint-scoped key first. If settings replacement
                # subsequently fails, the previous ordinary settings remain valid;
                # an extra endpoint key slot is harmless and recoverable.
                _atomic_write_json(self.keys_path, key_payload)
            _atomic_write_json(self.settings_path, settings_payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            safe = redact_text(exc, secrets=(settings.llm.api_key,))
            raise SettingsError(f"无法保存本地设置：{safe}") from None

    def _load_key_map(self) -> dict[str, str]:
        source_path = self.keys_path
        if (
            not source_path.exists()
            and self.legacy_keys_path is not None
            and self.legacy_keys_path != source_path
            and self.legacy_keys_path.exists()
        ):
            source_path = self.legacy_keys_path
        if not source_path.exists():
            return {}
        payload = _read_owner_only_json(source_path)
        if not isinstance(payload, Mapping):
            raise ValueError("API key 文件顶层必须是 JSON 对象")
        if payload.get("version") != KEYS_FORMAT_VERSION:
            raise ValueError("API key 文件版本不受支持")
        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, Mapping):
            raise ValueError("API key 文件缺少 keys 对象")
        result: dict[str, str] = {}
        for slot, value in raw_keys.items():
            if not isinstance(slot, str) or not isinstance(value, str):
                raise ValueError("API key 文件包含无效条目")
            if value:
                result[slot] = value
        return result


def default_app_settings() -> AppSettings:
    """Build defaults from environment-backed values in :mod:`config`."""

    return AppSettings(
        llm=LLMSettings(
            base_url=config.DEFAULT_BASE_URL or None,
            model=config.DEFAULT_MODEL,
            api_key=_environment_key(config.DEFAULT_BASE_URL),
            reasoning_effort=config.DEFAULT_REASONING_EFFORT,
        )
    )


def _endpoint_from_storage(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        raise ValueError("设置文件顶层必须是 JSON 对象")
    version = payload.get("version")
    if version not in {
        LEGACY_SETTINGS_FORMAT_VERSION,
        PREVIOUS_SETTINGS_FORMAT_VERSION,
        SETTINGS_FORMAT_VERSION,
    }:
        raise ValueError("设置文件版本不受支持")
    llm_payload = payload.get("llm")
    if not isinstance(llm_payload, Mapping):
        raise ValueError("设置文件缺少 llm 对象")
    raw_base_url = llm_payload.get("base_url")
    base_url = str(raw_base_url).strip().rstrip("/") if raw_base_url else None
    if base_url is None and version == LEGACY_SETTINGS_FORMAT_VERSION:
        provider = str(llm_payload.get("provider", "")).strip().lower()
        if provider == "deepseek":
            base_url = config.DEEPSEEK_BASE_URL
        elif provider == "openai":
            base_url = config.OPENAI_BASE_URL
    return base_url


def _settings_from_storage(payload: object, *, api_key: str) -> AppSettings:
    base_url = _endpoint_from_storage(payload)
    assert isinstance(payload, Mapping)
    llm_payload = payload["llm"]
    assert isinstance(llm_payload, Mapping)
    return AppSettings(
        version=SETTINGS_FORMAT_VERSION,
        sentence_difficulty=payload.get(
            "sentence_difficulty", SentenceDifficulty.CET6_POSTGRAD.value
        ),
        llm=LLMSettings(
            base_url=base_url,
            model=llm_payload.get("model", ""),
            api_key=api_key,
            reasoning_effort=llm_payload.get("reasoning_effort", "disabled"),
            timeout_seconds=float(llm_payload.get("timeout_seconds", 45.0)),
            max_retries=int(llm_payload.get("max_retries", 3)),
        ),
    )


def _key_slot(base_url: str | None) -> str:
    # Preserve path case: hostnames are case-insensitive, URL paths need not be.
    # Treating the full URL literally is safer than ever reusing a key across
    # two endpoints that only differ by a case-sensitive path.
    normalized_url = (base_url or "").strip().rstrip("/")
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    return f"endpoint:{digest}"


def _legacy_custom_slot(base_url: str | None) -> str:
    normalized_url = (base_url or "").strip().rstrip("/")
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    return f"custom:{digest}"


def _candidate_key_slots(base_url: str | None) -> tuple[str, ...]:
    normalized = (base_url or "").strip().rstrip("/")
    slots = [_key_slot(normalized), _legacy_custom_slot(normalized)]
    if normalized == config.DEEPSEEK_BASE_URL.rstrip("/"):
        slots.append("deepseek")
    if normalized == config.OPENAI_BASE_URL.rstrip("/"):
        slots.append("openai")
    return tuple(dict.fromkeys(slots))


def _environment_key(base_url: str | None) -> str:
    if config.API_KEY:
        return config.API_KEY
    normalized = (base_url or "").strip().rstrip("/")
    if normalized == config.DEEPSEEK_BASE_URL.rstrip("/"):
        return config.DEEPSEEK_API_KEY
    if normalized == config.OPENAI_BASE_URL.rstrip("/"):
        return config.OPENAI_API_KEY
    return ""


def _read_owner_only_json(path: Path) -> object:
    # Self-heal files created with an unsafe umask or by an older version.
    os.chmod(path, 0o600)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    _atomic_write(path, serialized + "\n")


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace *path* with a UTF-8 owner-only file."""

    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
    )
    temporary_path = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            file_descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
        _fsync_directory(parent)
    except Exception:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "ApiKeySource",
    "AppSettings",
    "KEYS_FORMAT_VERSION",
    "KeyAction",
    "SETTINGS_FORMAT_VERSION",
    "SettingsError",
    "SettingsStore",
    "default_app_settings",
]
