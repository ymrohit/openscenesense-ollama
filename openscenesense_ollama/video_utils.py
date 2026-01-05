from __future__ import annotations

import logging
from typing import Optional

import ffmpeg
import cv2

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str, fallback_duration: float = 0.0) -> float:
    """Return video duration in seconds using ffprobe or cv2 as fallback."""
    duration = _probe_duration_ffmpeg(video_path)
    if duration and duration > 0:
        return duration

    duration = _probe_duration_cv2(video_path)
    if duration and duration > 0:
        return duration

    return fallback_duration


def _probe_duration_ffmpeg(video_path: str) -> Optional[float]:
    try:
        info = ffmpeg.probe(video_path)
        value = float(info["format"]["duration"])
        return value if value > 0 else None
    except Exception as exc:
        logger.debug("ffprobe duration lookup failed: %s", exc)
        return None


def _probe_duration_cv2(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = 0.0

        if fps and fps > 0 and frame_count > 0:
            duration = frame_count / fps

        if duration <= 0 and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ret, _ = cap.read()
            if ret:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    duration = pos_ms / 1000.0

        return duration if duration > 0 else None
    finally:
        cap.release()
