import math

import pytest

from openscenesense_ollama import (
    DynamicFrameSelector,
    OllamaVideoAnalyzer,
    UniformFrameSelector,
)
from openscenesense_ollama.exceptions import ConfigurationError
from openscenesense_ollama.frame_selectors import _target_frame_count
from openscenesense_ollama.models import VideoMetadata


def test_dynamic_selector_finds_cuts_without_exceeding_budget(synthetic_video):
    selector = DynamicFrameSelector(
        scene_change_threshold=0.10,
        scene_scan_fps=2.0,
        min_scene_gap=0.5,
    )
    analyzer = OllamaVideoAnalyzer(
        min_frames=10,
        max_frames=10,
        frames_per_minute=60,
        frame_selector=selector,
        preflight=False,
    )

    frames = selector.select_frames(str(synthetic_video), analyzer)

    assert len(frames) == 10
    assert len({frame.timestamp for frame in frames}) == len(frames)
    cuts = [frame.timestamp for frame in frames if frame.selection_reason == "scene_change"]
    assert all(any(abs(found - expected) <= 0.55 for found in cuts) for expected in (1, 2, 3))
    assert selector.last_scan_frame_count <= math.ceil(4 * 2.0) + 2


@pytest.mark.parametrize("selector_type", [DynamicFrameSelector, UniformFrameSelector])
def test_selectors_handle_one_frame_video(one_frame_video, selector_type):
    selector = selector_type()
    analyzer = OllamaVideoAnalyzer(frame_selector=selector, preflight=False)

    frames = selector.select_frames(str(one_frame_video), analyzer)

    assert len(frames) == 1
    assert frames[0].selection_reason == "opening"


def test_slow_gradient_has_no_false_scene_changes(gradient_video):
    selector = DynamicFrameSelector(scene_change_threshold=0.18, scene_scan_fps=4)
    analyzer = OllamaVideoAnalyzer(
        frame_selector=selector,
        min_frames=8,
        max_frames=8,
        frames_per_minute=120,
        preflight=False,
    )

    frames = selector.select_frames(str(gradient_video), analyzer)

    assert all(frame.selection_reason != "scene_change" for frame in frames)


def test_zero_fps_scan_and_invalid_budget_configuration():
    selector = DynamicFrameSelector()
    assert selector._scan_indices(VideoMetadata(frame_count=10, duration=2, fps=0)) == [
        0,
        2,
        4,
        6,
        9,
    ]
    assert len(selector._scan_indices(VideoMetadata(frame_count=121, duration=0, fps=0))) == 120
    with pytest.raises(ConfigurationError):
        _target_frame_count(1, 1, 0, 1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scene_change_threshold": -0.1},
        {"scene_scan_fps": 0},
        {"scan_width": 31},
        {"scene_budget_ratio": 1.1},
    ],
)
def test_dynamic_selector_rejects_invalid_configuration(kwargs):
    with pytest.raises(ConfigurationError):
        DynamicFrameSelector(**kwargs)
