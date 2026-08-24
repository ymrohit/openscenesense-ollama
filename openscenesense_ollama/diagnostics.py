from __future__ import annotations

import shutil
import subprocess
from typing import Any


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def check_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "ok": False,
    }
    if report["ffmpeg"]:
        try:
            completed = subprocess.run(
                [report["ffmpeg"], "-version"],
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            )
            report["ffmpeg_version"] = completed.stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError) as exc:
            report["ffmpeg_error"] = str(exc)
    report["ok"] = bool(report["ffmpeg"] and report["ffprobe"])
    return report
