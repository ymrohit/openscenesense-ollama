"""Run the local Ollama integration against a generated two-scene video."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from jsonschema import validate

from openscenesense_ollama import (
    OllamaVideoAnalyzer,
    UniformFrameSelector,
    analysis_result_schema,
)


def _fixture(directory: str) -> Path:
    path = Path(directory) / "integration.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (96, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the integration fixture.")
    for color in ((0, 0, 0), (0, 0, 0), (255, 255, 255), (255, 255, 255)):
        writer.write(np.full((64, 96, 3), color, dtype=np.uint8))
    writer.release()
    return path


def main() -> None:
    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=os.environ.get("OLLAMA_VISION_MODEL") or "ministral-3:latest",
        summary_model=os.environ.get("OLLAMA_SUMMARY_MODEL") or "ministral-3:latest",
        host=os.environ.get("OLLAMA_HOST") or "http://localhost:11434",
        frame_selector=UniformFrameSelector(),
        min_frames=2,
        max_frames=2,
        frames_per_minute=60,
        strict=True,
    )
    with tempfile.TemporaryDirectory(prefix="openscenesense-ollama-integration-") as directory:
        result = analyzer.analyze_video_structured(str(_fixture(directory)))
    validate(result.to_dict(), analysis_result_schema())
    if len(result.frame_analyses) != 2 or not result.summary.brief:
        raise SystemExit("Ollama integration returned an incomplete result.")
    print(
        f"Ollama integration OK: frames={len(result.frame_analyses)}, "
        f"evaluated_tokens={result.metadata.usage.output_tokens}"
    )


if __name__ == "__main__":
    main()
