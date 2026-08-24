from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import cv2
import ffmpeg

from .exceptions import VideoLoadError
from .models import VideoMetadata

logger = logging.getLogger(__name__)


def _rate(value: object) -> float:
    try:
        return float(Fraction(str(value))) if value not in (None, "", "0/0") else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(video_path: str) -> VideoMetadata:
    if not Path(video_path).is_file():
        raise VideoLoadError(f"Video file does not exist: {video_path}")

    try:
        info = ffmpeg.probe(video_path)
        stream = next(item for item in info.get("streams", []) if item.get("codec_type") == "video")
        fps = _rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        frame_count = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
        if frame_count <= 0 and fps > 0 and duration > 0:
            frame_count = max(1, round(duration * fps))
        metadata = VideoMetadata(
            duration=max(0.0, duration),
            fps=max(0.0, fps),
            frame_count=max(0, frame_count),
            width=max(0, int(stream.get("width") or 0)),
            height=max(0, int(stream.get("height") or 0)),
        )
        if metadata.duration > 0 and metadata.frame_count > 0:
            return metadata
    except (ffmpeg.Error, OSError, StopIteration, TypeError, ValueError) as exc:
        logger.debug("FFprobe metadata lookup failed: %s", exc)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoLoadError(f"Failed to open video file: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        if duration <= 0 and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if cap.read()[0]:
                duration = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000)
        return VideoMetadata(
            duration=duration,
            fps=max(0.0, fps),
            frame_count=frame_count,
            width=max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)),
            height=max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)),
        )
    finally:
        cap.release()


def has_audio_stream(video_path: str) -> bool:
    try:
        info = ffmpeg.probe(video_path)
        return any(item.get("codec_type") == "audio" for item in info.get("streams", []))
    except (ffmpeg.Error, OSError, TypeError) as exc:
        logger.debug("FFprobe audio lookup failed: %s", exc)
        return True


def get_video_duration(video_path: str, fallback_duration: float = 0.0) -> float:
    duration = _probe_duration_ffmpeg(video_path) or _probe_duration_cv2(video_path)
    return duration if duration and duration > 0 else fallback_duration


def _probe_duration_ffmpeg(video_path: str) -> float | None:
    try:
        info = ffmpeg.probe(video_path)
        value = float(info.get("format", {}).get("duration") or 0)
        return value if value > 0 else None
    except Exception as exc:
        logger.debug("FFprobe duration lookup failed: %s", exc)
        return None


def _probe_duration_cv2(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        if duration <= 0 and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if cap.read()[0]:
                duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
        return duration if duration > 0 else None
    finally:
        cap.release()
