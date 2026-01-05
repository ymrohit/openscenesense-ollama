import openscenesense_ollama.video_utils as video_utils


def test_get_video_duration_prefers_ffmpeg(monkeypatch):
    def fake_probe(_path):
        return {"format": {"duration": "12.34"}}

    monkeypatch.setattr(video_utils.ffmpeg, "probe", fake_probe)
    assert video_utils.get_video_duration("dummy") == 12.34


def test_get_video_duration_fallback_cv2(monkeypatch):
    def fake_probe(_path):
        raise RuntimeError("ffprobe unavailable")

    class FakeCap:
        def __init__(self):
            self._pos = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == video_utils.cv2.CAP_PROP_FPS:
                return 25.0
            if prop == video_utils.cv2.CAP_PROP_FRAME_COUNT:
                return 250
            if prop == video_utils.cv2.CAP_PROP_POS_FRAMES:
                return self._pos
            return 0.0

        def set(self, prop, value):
            if prop == video_utils.cv2.CAP_PROP_POS_FRAMES:
                self._pos = int(value)
                return True
            return False

        def read(self):
            return True, None

        def release(self):
            return None

    monkeypatch.setattr(video_utils.ffmpeg, "probe", fake_probe)
    monkeypatch.setattr(video_utils.cv2, "VideoCapture", lambda _path: FakeCap())

    assert video_utils.get_video_duration("dummy") == 10.0


def test_get_video_duration_uses_fallback_when_unavailable(monkeypatch):
    def fake_probe(_path):
        raise RuntimeError("ffprobe unavailable")

    class ClosedCap:
        def isOpened(self):
            return False

        def release(self):
            return None

    monkeypatch.setattr(video_utils.ffmpeg, "probe", fake_probe)
    monkeypatch.setattr(video_utils.cv2, "VideoCapture", lambda _path: ClosedCap())

    assert video_utils.get_video_duration("dummy", fallback_duration=5.5) == 5.5
