from openscenesense_ollama.frame_selectors import _compute_frame_indices


def test_compute_frame_indices_empty():
    assert _compute_frame_indices(0, 5) == []
    assert _compute_frame_indices(5, 0) == []


def test_compute_frame_indices_single():
    assert _compute_frame_indices(1, 5) == [0]
    assert _compute_frame_indices(5, 1) == [0]


def test_compute_frame_indices_caps_to_total():
    assert _compute_frame_indices(3, 10) == [0, 1, 2]


def test_compute_frame_indices_spacing():
    assert _compute_frame_indices(5, 3) == [0, 2, 4]


def test_compute_frame_indices_monotonic():
    indices = _compute_frame_indices(7, 5)
    assert indices == sorted(indices)
    assert all(0 <= idx < 7 for idx in indices)
