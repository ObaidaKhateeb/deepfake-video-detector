"""
Fake Video Detector
Entry point — launches the PyQt5 GUI.

On first run, missing packages are installed automatically before startup.
ffmpeg is bundled via imageio-ffmpeg — no manual download or PATH setup needed.
"""

import importlib.util
import subprocess
import sys
import os

# ── Dependency bootstrap ──────────────────────────────────────────────────────

_CORE = {
    "PyQt5":             "PyQt5",
    "opencv-python":     "cv2",
    "numpy":             "numpy",
    "anthropic":         "anthropic",
    "ddgs":              "ddgs",
    "imageio-ffmpeg":    "imageio_ffmpeg",
}

_OPTIONAL = {"faster-whisper": "faster_whisper"}


def _missing(packages: dict) -> list:
    return [pip for pip, imp in packages.items()
            if importlib.util.find_spec(imp) is None]


def _install(packages: list) -> None:
    print(f"[setup] Installing: {', '.join(packages)} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        check=True,
    )
    print("[setup] Done.")


def _bootstrap() -> None:
    missing_core     = _missing(_CORE)
    missing_optional = _missing(_OPTIONAL)

    if not missing_core and not missing_optional:
        return

    if missing_core:
        print(f"[setup] Missing required packages: {', '.join(missing_core)}")
        _install(missing_core)

    if missing_optional:
        print(f"[setup] Missing optional packages: {', '.join(missing_optional)}")
        _install(missing_optional)

    # Restart so every newly installed package is importable cleanly
    print("[setup] Restarting ...")
    subprocess.run([sys.executable] + sys.argv)
    sys.exit(0)


_bootstrap()

# ── Inject bundled ffmpeg into PATH ──────────────────────────────────────────
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# ── Normal startup ────────────────────────────────────────────────────────────

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Fake Video Detector")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
