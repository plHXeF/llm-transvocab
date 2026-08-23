"""PyInstaller entrypoint for the one-click desktop application."""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap

from desktop_runtime import bundle_root, configure_desktop_environment


def available_port() -> int:
    """Reserve a currently available loopback port for the local server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def streamlit_options(port: int) -> dict[str, object]:
    """Return deterministic options suitable for a bundled local app."""

    return {
        "global.developmentMode": False,
        "server.address": "127.0.0.1",
        "server.port": port,
        "server.headless": True,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
        "logger.level": "info",
    }


def _open_when_ready(url: str, *, attempts: int = 120) -> None:
    health_url = f"{url}/_stcore/health"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)


def run_streamlit(script: Path, options: dict[str, object]) -> None:
    """Load options in the same order as Streamlit's command-line entrypoint."""

    bootstrap.load_config_options(flag_options=options)
    bootstrap.run(str(script), False, [], options)


def main() -> None:
    configure_desktop_environment()
    script = bundle_root() / "vocab_web.py"
    if not script.is_file():
        raise FileNotFoundError(f"缺少应用入口：{script.name}")

    port = available_port()
    url = f"http://127.0.0.1:{port}"
    threading.Thread(
        target=_open_when_ready,
        args=(url,),
        daemon=True,
        name="desktop-browser-opener",
    ).start()
    run_streamlit(script, streamlit_options(port))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
