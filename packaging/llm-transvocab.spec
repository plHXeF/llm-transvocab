# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


ROOT = Path(SPECPATH).parent
APP_NAME = "LLM TransVocab"
PROJECT_MODULES = [
    "app_settings",
    "config",
    "desktop_runtime",
    "domain",
    "learning_charts",
    "learning_store",
    "llm_service",
    "model_error_log",
    "prefetch",
    "scheduler",
    "vocabulary_repository",
]

datas = [
    (str(ROOT / "vocab_web.py"), "."),
    (str(ROOT / "vocabularies.csv"), "."),
]
datas += collect_data_files("streamlit")
datas += collect_data_files("plotly")
for distribution in ("streamlit", "altair", "openai", "plotly"):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

hiddenimports = PROJECT_MODULES + collect_submodules("streamlit")

a = Analysis(
    [str(ROOT / "desktop_launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
collection = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="com.plhxef.llm-transvocab",
        info_plist={
            "CFBundleDisplayName": APP_NAME,
            "NSHighResolutionCapable": True,
        },
    )
