import json

import pytest
from jsonschema import validate

from openscenesense_ollama import OllamaVideoAnalyzer, UniformFrameSelector
from openscenesense_ollama.exceptions import ResponseValidationError
from openscenesense_ollama.models import analysis_result_schema


class DummyResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def _summary_response():
    return DummyResponse(
        {
            "message": {
                "content": json.dumps(
                    {
                        "detailed": "Detailed local summary",
                        "brief": "Brief local summary",
                        "events": [
                            {
                                "start_time": 0,
                                "end_time": 1,
                                "description": "An event",
                                "source_frame_timestamps": [0],
                                "objects": [],
                                "actions": [],
                                "visible_text": [],
                            }
                        ],
                    }
                )
            },
            "prompt_eval_count": 8,
            "eval_count": 4,
        }
    )


def test_full_structured_pipeline_with_custom_processor(synthetic_video, monkeypatch):
    progress = []
    analyzer = OllamaVideoAnalyzer(
        frame_selector=UniformFrameSelector(),
        min_frames=3,
        max_frames=3,
        frames_per_minute=3,
        custom_frame_processor=lambda frame: {
            "description": f"frame {frame.timestamp:.2f}",
            "objects": ["frame"],
        },
        on_progress=progress.append,
        preflight=False,
    )
    calls = []

    def fake_post(payload, label, endpoint=None):
        calls.append((payload, label, endpoint))
        return _summary_response()

    monkeypatch.setattr(analyzer, "_post_with_retries", fake_post)

    result = analyzer.analyze_video_structured(str(synthetic_video))

    validate(result.to_dict(), analysis_result_schema())
    assert len(result.frame_analyses) == 3
    assert result.summary.brief == "Brief local summary"
    assert result.timeline[0].description == "An event"
    assert result.metadata.usage.total_tokens == 12
    assert calls[0][1] == "summary"
    assert progress[-1].stage == "complete"
    assert result.to_legacy_dict()["brief_summary"] == "Brief local summary"


def test_resume_reuses_custom_frame_results(synthetic_video, tmp_path, monkeypatch):
    calls = {"frames": 0}

    def processor(frame):
        calls["frames"] += 1
        return {"description": f"frame {frame.timestamp:.2f}"}

    def run():
        analyzer = OllamaVideoAnalyzer(
            frame_selector=UniformFrameSelector(),
            min_frames=3,
            max_frames=3,
            frames_per_minute=3,
            custom_frame_processor=processor,
            preflight=False,
            cache_dir=str(tmp_path),
            resume=True,
        )
        monkeypatch.setattr(
            analyzer, "_post_with_retries", lambda *_args, **_kwargs: _summary_response()
        )
        return analyzer.analyze_video_structured(str(synthetic_video))

    run()
    assert calls["frames"] == 3
    run()
    assert calls["frames"] == 3


def test_independent_context_uses_worker_pipeline(synthetic_video, monkeypatch):
    analyzer = OllamaVideoAnalyzer(
        frame_selector=UniformFrameSelector(),
        min_frames=4,
        max_frames=4,
        frames_per_minute=4,
        custom_frame_processor=lambda frame: {"description": str(frame.timestamp)},
        context_mode="independent",
        max_workers=2,
        preflight=False,
    )
    monkeypatch.setattr(
        analyzer, "_post_with_retries", lambda *_args, **_kwargs: _summary_response()
    )

    result = analyzer.analyze_video_structured(str(synthetic_video))

    assert len(result.frame_analyses) == 4
    assert [item.timestamp for item in result.frame_analyses] == sorted(
        item.timestamp for item in result.frame_analyses
    )


def test_partial_custom_processor_failures_obey_ratio(synthetic_video, monkeypatch):
    def fail(_frame):
        raise RuntimeError("processor failed")

    analyzer = OllamaVideoAnalyzer(
        frame_selector=UniformFrameSelector(),
        min_frames=2,
        max_frames=2,
        frames_per_minute=2,
        custom_frame_processor=fail,
        max_frame_failure_ratio=1,
        preflight=False,
    )
    monkeypatch.setattr(
        analyzer, "_post_with_retries", lambda *_args, **_kwargs: _summary_response()
    )

    result = analyzer.analyze_video_structured(str(synthetic_video))

    assert result.metadata.failed_frame_analyses == 2
    assert result.errors == ["processor failed", "processor failed"]

    analyzer.max_frame_failure_ratio = 0.25
    with pytest.raises(ResponseValidationError, match="failure ratio"):
        analyzer.analyze_video_structured(str(synthetic_video))


def test_strict_custom_processor_failure_raises(synthetic_video):
    def fail(_frame):
        raise RuntimeError("processor failed")

    analyzer = OllamaVideoAnalyzer(
        frame_selector=UniformFrameSelector(),
        min_frames=2,
        max_frames=2,
        frames_per_minute=2,
        custom_frame_processor=fail,
        strict=True,
        preflight=False,
    )

    with pytest.raises(RuntimeError, match="processor failed"):
        analyzer.analyze_video_structured(str(synthetic_video))


def test_malformed_summary_repairs_once_then_falls_back(synthetic_video, monkeypatch):
    analyzer = OllamaVideoAnalyzer(
        frame_selector=UniformFrameSelector(),
        min_frames=2,
        max_frames=2,
        frames_per_minute=2,
        custom_frame_processor=lambda _frame: {"description": "frame"},
        preflight=False,
    )
    calls = []

    def malformed(_payload, label, **_kwargs):
        calls.append(label)
        return DummyResponse({"message": {"content": "not json"}})

    monkeypatch.setattr(analyzer, "_post_with_retries", malformed)

    result = analyzer.analyze_video_structured(str(synthetic_video))

    assert calls == ["summary", "summary repair"]
    assert result.summary.detailed == "frame frame"
    assert any("deterministic fallback" in warning for warning in result.warnings)
