import pytest

from openscenesense_ollama.cache import AnalysisCache, analysis_cache_key


def test_cache_key_changes_with_video_content(tmp_path):
    video = tmp_path / "video.bin"
    video.write_bytes(b"a" * 100)
    first = analysis_cache_key(str(video), {"model": "one"})
    video.write_bytes(b"b" * 100)
    second = analysis_cache_key(str(video), {"model": "one"})
    assert first != second


def test_cache_json_writes_are_readable(tmp_path):
    cache = AnalysisCache(tmp_path, "key")
    cache.write_json("frame_analyses/0000.json", {"ok": True})
    assert cache.read_json("frame_analyses/0000.json") == {"ok": True}


def test_corrupt_cache_is_ignored_and_escape_is_blocked(tmp_path):
    cache = AnalysisCache(tmp_path, "key")
    result = cache.path("result.json")
    result.parent.mkdir(parents=True)
    result.write_text("{broken", encoding="utf-8")

    assert cache.read_json("result.json") is None
    with pytest.raises(ValueError, match="escapes"):
        cache.path("../outside.json")
