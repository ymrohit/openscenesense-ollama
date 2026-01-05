# frame_selector.py
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from abc import ABC, abstractmethod
from .models import Frame, SceneType
from .exceptions import VideoLoadError


def _compute_frame_indices(total_frames: int, target_frames: int) -> List[int]:
    if total_frames <= 0 or target_frames <= 0:
        return []

    count = min(total_frames, target_frames)
    indices = np.linspace(0, total_frames - 1, num=count, dtype=int)
    return sorted(set(indices.tolist()))


def _estimate_duration(cap: cv2.VideoCapture, fps: float, total_frames: int) -> float:
    duration = 0.0
    if fps and fps > 0 and total_frames > 0:
        duration = total_frames / fps

    if duration <= 0 and total_frames > 0:
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, _ = cap.read()
        if ret:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                duration = pos_ms / 1000.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    return duration

class FrameSelector(ABC):
    """Abstract base class for frame selection strategies"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def select_frames(self, video_path: str, analyzer: 'OllamaVideoAnalyzer') -> List[Frame]:
        pass

    def _validate_video(self, video_path: str) -> Tuple[cv2.VideoCapture, float, float, int]:
        """Validate video file and return video properties"""
        self.logger.info(f"Validating video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise VideoLoadError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = _estimate_duration(cap, fps, total_frames)

        if duration <= 0:
            self.logger.warning(
                "Video duration could not be determined reliably (fps=%s, total_frames=%s)",
                fps,
                total_frames,
            )

        self.logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        return cap, fps, duration, total_frames


class DynamicFrameSelector(FrameSelector):
    """Dynamic frame selection strategy based on scene changes and motion"""
    def __init__(self, threshold: float = 20.0):
        super().__init__()
        self.threshold = threshold

    def select_frames(self, video_path: str, analyzer: 'OllamaVideoAnalyzer') -> List[Frame]:
        self.logger.info("Starting dynamic frame selection")
        cap, fps, duration, total_frames = self._validate_video(video_path)

        scene_changes = self._detect_scene_changes(video_path, cap)
        target_frames = analyzer._calculate_dynamic_frame_count(duration, scene_changes)

        self.logger.info(f"Target frames for analysis: {target_frames}")
        frames = self._extract_frames(cap, target_frames, scene_changes)

        cap.release()
        self.logger.info(f"Dynamic frame selection completed. Selected {len(frames)} frames")
        return frames

    def _detect_scene_changes(self, video_path: str, cap: cv2.VideoCapture) -> List[float]:
        """Detect significant scene changes in the video"""
        self.logger.info("Detecting scene changes")
        scene_changes = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if prev_frame is not None:
                diff_score = self._calculate_frame_difference(prev_frame, frame)

                if diff_score > self.threshold:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    scene_changes.append(timestamp)

            prev_frame = frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return scene_changes

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff)

    def _extract_frames(self, cap: cv2.VideoCapture, target_frames: int, scene_changes: List[float]) -> List[Frame]:
        """Extract frames based on scene changes and target count"""
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = _compute_frame_indices(total_frames, target_frames)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to read frame at position %s", frame_idx)
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            is_scene_change = any(abs(sc - timestamp) < 0.1 for sc in scene_changes)
            scene_type = SceneType.TRANSITION if is_scene_change else SceneType.STATIC

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=scene_type
            ))

        return frames

class UniformFrameSelector(FrameSelector):
    """Uniform frame selection strategy selecting frames at regular intervals"""

    def select_frames(self, video_path: str, analyzer: 'OllamaVideoAnalyzer') -> List[Frame]:
        self.logger.info("Starting uniform frame selection")
        cap, fps, duration, total_frames = self._validate_video(video_path)

        target_frames = analyzer._calculate_uniform_frame_count(duration)
        self.logger.info(f"Target frames for uniform selection: {target_frames}")
        frames = self._extract_uniform_frames(cap, target_frames, fps)

        cap.release()
        self.logger.info(f"Uniform frame selection completed. Selected {len(frames)} frames")
        return frames

    def _extract_uniform_frames(self, cap: cv2.VideoCapture, target_frames: int, fps: float) -> List[Frame]:
        """Extract frames at uniform intervals"""
        frames = []
        if target_frames <= 0:
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = _compute_frame_indices(total_frames, target_frames)
        self.logger.debug("Frame extraction indices: %s", frame_indices)

        for frame_number in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(f"Failed to read frame at position {frame_number}")
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=SceneType.STATIC
            ))

        return frames

class AllFrameSelector(FrameSelector):
    """Frame selection strategy that selects all frames from the video"""

    def select_frames(self, video_path: str, analyzer: 'OllamaVideoAnalyzer') -> List[Frame]:
        self.logger.info("Starting all frame selection")
        cap, fps, duration, total_frames = self._validate_video(video_path)

        frames = self._extract_all_frames(cap, fps)
        cap.release()
        self.logger.info(f"All frame selection completed. Selected {len(frames)} frames")
        return frames

    def _extract_all_frames(self, cap: cv2.VideoCapture, fps: float) -> List[Frame]:
        """Extract all frames from the video"""
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=SceneType.STATIC  # Or determine dynamically if needed
            ))

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.logger.debug(f"Extracted {frame_idx} frames")

        return frames
