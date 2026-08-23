"""Desktop-bundle paths and first-run resource installation.

The packaged application is read-only.  Mutable vocabulary and learning data
therefore live in the current user's application-data directory.  Source-mode
launches keep using the repository paths defined by :mod:`config`.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping


APP_DIRECTORY_NAME = "LLM TransVocab"


@dataclass(frozen=True)
class DesktopPaths:
    """Writable paths used by an installed desktop build."""

    data_dir: Path
    vocabulary: Path
    settings: Path
    api_keys: Path
    learning_db: Path
    model_error_log: Path


def bundle_root() -> Path:
    """Return the directory containing bundled data files."""

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent


def user_data_directory(
    *,
    platform: str | None = None,
    environ: MutableMapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the conventional per-user writable directory for a platform."""

    platform_name = platform or sys.platform
    environment = os.environ if environ is None else environ
    home_directory = Path.home() if home is None else Path(home)

    if platform_name == "darwin":
        return home_directory / "Library" / "Application Support" / APP_DIRECTORY_NAME
    if platform_name.startswith("win"):
        local_app_data = environment.get("LOCALAPPDATA", "").strip()
        parent = (
            Path(local_app_data).expanduser()
            if local_app_data
            else home_directory / "AppData" / "Local"
        )
        return parent / APP_DIRECTORY_NAME

    xdg_data_home = environment.get("XDG_DATA_HOME", "").strip()
    parent = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else home_directory / ".local" / "share"
    )
    return parent / APP_DIRECTORY_NAME


def _copy_default_vocabulary(source: Path, destination: Path) -> None:
    """Install the bundled vocabulary once without overwriting user changes."""

    if destination.exists():
        return
    if not source.is_file():
        raise FileNotFoundError(f"缺少内置词库：{source.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            with source.open("rb") as bundled_file:
                shutil.copyfileobj(bundled_file, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def configure_desktop_environment(
    *,
    environ: MutableMapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
    bundled_directory: Path | None = None,
) -> DesktopPaths:
    """Prepare writable data and expose it through the existing env contract."""

    environment = os.environ if environ is None else environ
    data_dir = user_data_directory(
        platform=platform,
        environ=environment,
        home=home,
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = Path(
        environment.get("VOCAB_FILE", str(data_dir / "vocabularies.csv"))
    ).expanduser()
    paths = DesktopPaths(
        data_dir=data_dir,
        vocabulary=vocabulary,
        settings=Path(
            environment.get(
                "VOCAB_SETTINGS_FILE", str(data_dir / "app_settings.json")
            )
        ).expanduser(),
        api_keys=Path(
            environment.get(
                "VOCAB_API_KEYS_FILE", str(data_dir / "api_keys.json")
            )
        ).expanduser(),
        learning_db=Path(
            environment.get(
                "VOCAB_LEARNING_DB_FILE", str(data_dir / "learning.db")
            )
        ).expanduser(),
        model_error_log=Path(
            environment.get(
                "VOCAB_MODEL_ERROR_LOG_FILE",
                str(data_dir / "model_errors.jsonl"),
            )
        ).expanduser(),
    )

    source_root = bundle_root() if bundled_directory is None else bundled_directory
    _copy_default_vocabulary(source_root / "vocabularies.csv", paths.vocabulary)

    environment.setdefault("VOCAB_FILE", str(paths.vocabulary))
    environment.setdefault("VOCAB_SETTINGS_FILE", str(paths.settings))
    environment.setdefault("VOCAB_API_KEYS_FILE", str(paths.api_keys))
    environment.setdefault("VOCAB_LEARNING_DB_FILE", str(paths.learning_db))
    environment.setdefault("VOCAB_MODEL_ERROR_LOG_FILE", str(paths.model_error_log))
    environment["VOCAB_DESKTOP_MODE"] = "1"
    return paths
