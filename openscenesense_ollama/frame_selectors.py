from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .exceptions import ConfigurationError, VideoLoadError, VideoMetadataError
from .models import Frame, SceneType, SelectionMetadata, SelectionReason, VideoMetadata
from .video_utils import probe_video

if TYPE_CHECKING:
    from .analyzer import OllamaVideoAnalyzer


def _compute_frame_indices(total_frames: int, target_frames: int) -> list[int]:
    if total_frames <= 0 or target_frames <= 0:
        return []
    count = min(total_frames, target_frames)
    return sorted(set(np.linspace(0, total_frames - 1, num=count, dtype=int).tolist()))


def _target_frame_count(
    duration: float,
    total_frames: int,
    min_frames: int,
    max_frames: int,
    frames_per_minute: float,
) -> int:
    if min_frames < 1 or max_frames < min_frames or frames_per_minute <= 0:
        raise ConfigurationError("Invalid frame-budget configuration.")
    calculated = round(max(0.0, duration) / 60 * frames_per_minute)
    target = min(max_frames, max(min_frames, calculated))
    return min(total_frames, target) if total_frames > 0 else target


@dataclass(frozen=True)
class _Candidate:
    frame_index: int
    timestamp: float
    score: float


class FrameSelector(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_metadata = VideoMetadata()
        self.last_selection_metadata = SelectionMetadata()

    @abstractmethod
    def select_frames(self, video_path: str, analyzer: OllamaVideoAnalyzer) -> list[Frame]:
        raise NotImplementedError

    def selection_parameters(self) -> dict[str, object]:
        return {}

    @staticmethod
    def _open(video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoLoadError(f"Failed to open video file: {video_path}")
        return cap

    @staticmethod
    def _read_frame(
        cap: cv2.VideoCapture, frame_index: int, fps: float
    ) -> tuple[np.ndarray, float]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = cap.read()
        if not ok or image is None:
            raise VideoLoadError(f"Failed to decode frame {frame_index}.")
        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
        if timestamp <= 0 and fps > 0:
            timestamp = frame_index / fps
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), max(0.0, timestamp)

    @staticmethod
    def _iter_frames(
        cap: cv2.VideoCapture, frame_indices: list[int], fps: float
    ) -> Iterator[tuple[int, np.ndarray, float]]:
        """Yield sorted indices in one forward pass with constant image memory."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        position = 0
        for frame_index in sorted(set(frame_indices)):
            ok = True
            while position <= frame_index:
                ok = cap.grab()
                position += 1
                if not ok:
                    break
            if not ok:
                break
            ok, image = cap.retrieve()
            if not ok or image is None:
                continue
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
            if timestamp <= 0 and fps > 0:
                timestamp = frame_index / fps
            yield (
                frame_index,
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                max(0.0, timestamp),
            )

    @staticmethod
    def _read_frames(
        cap: cv2.VideoCapture, frame_indices: list[int], fps: float
    ) -> dict[int, tuple[np.ndarray, float]]:
        """Read a bounded final selection into memory."""
        return {
            frame_index: (image, timestamp)
            for frame_index, image, timestamp in FrameSelector._iter_frames(cap, frame_indices, fps)
        }


class DynamicFrameSelector(FrameSelector):
    def __init__(
        self,
        threshold: float | None = None,
        *,
        scene_change_threshold: float = 0.18,
        scene_scan_fps: float = 2.0,
        scan_width: int = 320,
        min_scene_gap: float = 0.75,
        scene_budget_ratio: float = 0.60,
    ) -> None:
        super().__init__()
        if threshold is not None:
            scene_change_threshold = threshold / 255 if threshold > 1 else threshold
        if not 0 <= scene_change_threshold <= 1:
            raise ConfigurationError("scene_change_threshold must be between 0 and 1.")
        if scene_scan_fps <= 0 or scan_width < 32 or min_scene_gap < 0:
            raise ConfigurationError("Invalid dynamic frame-selector configuration.")
        if not 0 <= scene_budget_ratio <= 1:
            raise ConfigurationError("scene_budget_ratio must be between 0 and 1.")
        self.scene_change_threshold = float(scene_change_threshold)
        self.scene_scan_fps = float(scene_scan_fps)
        self.scan_width = int(scan_width)
        self.min_scene_gap = float(min_scene_gap)
        self.scene_budget_ratio = float(scene_budget_ratio)
        self.last_scan_frame_count = 0

    @property
    def threshold(self) -> float:
        return self.scene_change_threshold * 255

    def selection_parameters(self) -> dict[str, object]:
        return {
            "scene_change_threshold": self.scene_change_threshold,
            "scene_scan_fps": self.scene_scan_fps,
            "scan_width": self.scan_width,
            "min_scene_gap": self.min_scene_gap,
            "scene_budget_ratio": self.scene_budget_ratio,
        }

    def _scan_indices(self, metadata: VideoMetadata) -> list[int]:
        if metadata.frame_count <= 0:
            return []
        if metadata.fps > 0:
            step = max(1, round(metadata.fps / self.scene_scan_fps))
            indices = list(range(0, metadata.frame_count, step))
            if indices[-1] != metadata.frame_count - 1:
                indices.append(metadata.frame_count - 1)
            return indices
        if metadata.duration > 0:
            count = max(2, math.ceil(metadata.duration * self.scene_scan_fps) + 1)
            return _compute_frame_indices(metadata.frame_count, count)
        return _compute_frame_indices(metadata.frame_count, min(metadata.frame_count, 120))

    @staticmethod
    def _calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        return float(np.mean(cv2.absdiff(gray1, gray2)) / 255)

    def _detect_scene_changes(
        self, _video_path: str, cap: cv2.VideoCapture, metadata: VideoMetadata | None = None
    ) -> list[_Candidate]:
        metadata = metadata or self.last_metadata
        samples: list[_Candidate] = []
        previous_gray: np.ndarray | None = None
        scan_indices = self._scan_indices(metadata)
        for frame_index, frame, timestamp in self._iter_frames(cap, scan_indices, metadata.fps):
            height, width = frame.shape[:2]
            if width > self.scan_width:
                scaled_height = max(1, round(height * self.scan_width / width))
                frame = cv2.resize(
                    frame, (self.scan_width, scaled_height), interpolation=cv2.INTER_AREA
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if previous_gray is not None:
                score = float(np.mean(cv2.absdiff(previous_gray, gray)) / 255)
                samples.append(_Candidate(frame_index, timestamp, score))
            previous_gray = gray

        self.last_scan_frame_count = len(samples) + (1 if previous_gray is not None else 0)
        peaks = []
        for index, candidate in enumerate(samples):
            previous_score = samples[index - 1].score if index else -1.0
            next_score = samples[index + 1].score if index + 1 < len(samples) else -1.0
            if (
                candidate.score >= self.scene_change_threshold
                and candidate.score >= previous_score
                and candidate.score >= next_score
            ):
                peaks.append(candidate)

        retained: list[_Candidate] = []
        for candidate in sorted(peaks, key=lambda item: item.score, reverse=True):
            if all(
                abs(candidate.timestamp - item.timestamp) >= self.min_scene_gap for item in retained
            ):
                retained.append(candidate)
        return sorted(retained, key=lambda item: item.timestamp)

    @staticmethod
    def _fill_largest_gaps(selected: set[int], total_frames: int, budget: int) -> None:
        while len(selected) < budget:
            ordered = sorted(selected)
            best: tuple[int, int] | None = None
            for left, right in zip(ordered, ordered[1:], strict=False):
                if right - left > 1 and (best is None or right - left > best[1] - best[0]):
                    best = (left, right)
            if best is None:
                for candidate in range(total_frames):
                    if candidate not in selected:
                        selected.add(candidate)
                        break
                else:
                    return
            else:
                selected.add((best[0] + best[1]) // 2)

    def select_frames(self, video_path: str, analyzer: OllamaVideoAnalyzer) -> list[Frame]:
        metadata = probe_video(video_path)
        self.last_metadata = metadata
        if metadata.frame_count <= 0:
            raise VideoMetadataError("Video frame count could not be determined.")
        target = _target_frame_count(
            metadata.duration,
            metadata.frame_count,
            analyzer.min_frames,
            analyzer.max_frames,
            analyzer.frames_per_minute,
        )
        cap = self._open(video_path)
        try:
            scene_changes = self._detect_scene_changes(video_path, cap, metadata)
            selected: dict[int, tuple[str, float]] = {}
            if target:
                selected[0] = (SelectionReason.OPENING.value, 0.0)
            if target > 1 and metadata.frame_count > 1:
                selected[metadata.frame_count - 1] = (SelectionReason.CLOSING.value, 0.0)
            remaining = max(0, target - len(selected))
            scene_slots = min(len(scene_changes), math.floor(remaining * self.scene_budget_ratio))
            for candidate in sorted(scene_changes, key=lambda item: item.score, reverse=True)[
                :scene_slots
            ]:
                selected.setdefault(
                    candidate.frame_index,
                    (SelectionReason.SCENE_CHANGE.value, candidate.score),
                )
            indices = set(selected)
            self._fill_largest_gaps(indices, metadata.frame_count, target)
            for index in indices:
                selected.setdefault(index, (SelectionReason.UNIFORM_FILL.value, 0.0))

            frames: list[Frame] = []
            cap.release()
            cap = self._open(video_path)
            decoded = self._read_frames(cap, sorted(selected)[:target], metadata.fps)
            for frame_index in sorted(selected)[:target]:
                if frame_index not in decoded:
                    self.logger.warning("Failed to decode frame %s", frame_index)
                    continue
                image, timestamp = decoded[frame_index]
                reason, score = selected[frame_index]
                frames.append(
                    Frame(
                        image=image,
                        timestamp=timestamp,
                        scene_type=(
                            SceneType.TRANSITION
                            if reason == SelectionReason.SCENE_CHANGE.value
                            else SceneType.STATIC
                        ),
                        difference_score=score,
                        selection_reason=reason,
                    )
                )
        finally:
            cap.release()
        self.last_selection_metadata = SelectionMetadata(
            strategy="dynamic",
            selected_frame_count=len(frames),
            parameters={
                "min_frames": analyzer.min_frames,
                "max_frames": analyzer.max_frames,
                "frames_per_minute": analyzer.frames_per_minute,
                **self.selection_parameters(),
                "scanned_frame_count": self.last_scan_frame_count,
            },
        )
        return frames


class UniformFrameSelector(FrameSelector):
    def select_frames(self, video_path: str, analyzer: OllamaVideoAnalyzer) -> list[Frame]:
        metadata = probe_video(video_path)
        if metadata.frame_count <= 0:
            raise VideoMetadataError("Video frame count could not be determined.")
        target = _target_frame_count(
            metadata.duration,
            metadata.frame_count,
            analyzer.min_frames,
            analyzer.max_frames,
            analyzer.frames_per_minute,
        )
        indices = _compute_frame_indices(metadata.frame_count, target)
        cap = self._open(video_path)
        frames: list[Frame] = []
        try:
            decoded = self._read_frames(cap, indices, metadata.fps)
            for position, frame_index in enumerate(indices):
                if frame_index not in decoded:
                    self.logger.warning("Failed to decode frame %s", frame_index)
                    continue
                image, timestamp = decoded[frame_index]
                reason = SelectionReason.UNIFORM_FILL.value
                if position == 0:
                    reason = SelectionReason.OPENING.value
                elif position == len(indices) - 1:
                    reason = SelectionReason.CLOSING.value
                frames.append(
                    Frame(
                        image=image,
                        timestamp=timestamp,
                        scene_type=SceneType.STATIC,
                        selection_reason=reason,
                    )
                )
        finally:
            cap.release()
        self.last_metadata = metadata
        self.last_selection_metadata = SelectionMetadata(
            strategy="uniform",
            selected_frame_count=len(frames),
            parameters={
                "min_frames": analyzer.min_frames,
                "max_frames": analyzer.max_frames,
                "frames_per_minute": analyzer.frames_per_minute,
            },
        )
        return frames


class AllFrameSelector(FrameSelector):
    """Compatibility selector that intentionally decodes every frame."""

    def select_frames(self, video_path: str, analyzer: OllamaVideoAnalyzer) -> list[Frame]:
        metadata = probe_video(video_path)
        cap = self._open(video_path)
        frames: list[Frame] = []
        try:
            frame_index = 0
            while True:
                ok, image = cap.read()
                if not ok or image is None:
                    break
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
                if timestamp <= 0 and metadata.fps > 0:
                    timestamp = frame_index / metadata.fps
                frames.append(
                    Frame(
                        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        timestamp=max(0.0, timestamp),
                        scene_type=SceneType.STATIC,
                        selection_reason=(
                            SelectionReason.OPENING.value
                            if frame_index == 0
                            else SelectionReason.UNIFORM_FILL.value
                        ),
                    )
                )
                frame_index += 1
        finally:
            cap.release()
        if len(frames) > 1:
            frames[-1].selection_reason = SelectionReason.CLOSING.value
        self.last_metadata = metadata
        self.last_selection_metadata = SelectionMetadata(
            strategy="all", selected_frame_count=len(frames), parameters={}
        )
        return frames
