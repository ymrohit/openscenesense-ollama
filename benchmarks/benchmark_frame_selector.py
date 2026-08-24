"""Compare the v1.2 reduced-rate selector with the v1.1 full-frame scan."""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from openscenesense_ollama import DynamicFrameSelector, OllamaVideoAnalyzer
from openscenesense_ollama.video_utils import probe_video


def legacy_full_frame_scan(video_path: str) -> tuple[float, int]:
    capture = cv2.VideoCapture(video_path)
    previous = None
    inspected = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                np.mean(cv2.absdiff(previous, gray))
            previous = gray
            inspected += 1
    finally:
        capture.release()
    return time.perf_counter() - started, inspected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--scene-scan-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()

    metadata = probe_video(args.video_path)
    legacy_seconds, legacy_frames = legacy_full_frame_scan(args.video_path)
    selector = DynamicFrameSelector(scene_scan_fps=args.scene_scan_fps)
    analyzer = OllamaVideoAnalyzer(
        min_frames=8,
        max_frames=args.max_frames,
        frames_per_minute=4,
        frame_selector=selector,
        preflight=False,
    )
    started = time.perf_counter()
    selected = selector.select_frames(args.video_path, analyzer)
    modern_seconds = time.perf_counter() - started
    report = {
        "duration": metadata.duration,
        "legacy_seconds": legacy_seconds,
        "legacy_inspected_frames": legacy_frames,
        "v1_2_seconds": modern_seconds,
        "v1_2_scanned_frames": selector.last_scan_frame_count,
        "selected_frames": len(selected),
        "inspection_reduction": (
            1 - selector.last_scan_frame_count / legacy_frames if legacy_frames else 0
        ),
        "speedup": legacy_seconds / modern_seconds if modern_seconds else None,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
