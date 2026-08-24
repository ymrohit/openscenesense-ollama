from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "cuts.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (160, 90),
    )
    assert writer.isOpened()
    for color in ((0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0)):
        for _ in range(12):
            writer.write(np.full((90, 160, 3), color, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def one_frame_video(tmp_path):
    path = tmp_path / "one-frame.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (64, 48),
    )
    assert writer.isOpened()
    writer.write(np.full((48, 64, 3), 127, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def gradient_video(tmp_path):
    path = tmp_path / "gradient.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (96, 64),
    )
    assert writer.isOpened()
    for value in range(48):
        writer.write(np.full((64, 96, 3), value * 3, dtype=np.uint8))
    writer.release()
    return path
