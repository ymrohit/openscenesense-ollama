import base64
import json

import numpy as np
import pytest

from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.exceptions import (
    ConfigurationError,
    ModelCapabilityError,
    ModelNotFoundError,
)
from openscenesense_ollama.models import Frame, FrameAnalysis, SceneType


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_preflight_checks_version_models_and_vision(monkeypatch):
    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model="vision:latest",
        summary_model="summary:latest",
        request_retries=0,
    )

    def fake_get(endpoint, **_kwargs):
        if endpoint.endswith("/api/version"):
            return DummyResponse({"version": "1.0.0"})
        return DummyResponse({"models": [{"name": "vision:latest"}, {"name": "summary:latest"}]})

    monkeypatch.setattr(analyzer.session, "get", fake_get)
    monkeypatch.setattr(
        analyzer.session,
        "post",
        lambda *_args, **_kwargs: DummyResponse({"capabilities": ["completion", "vision"]}),
    )

    report = analyzer.check_environment()

    assert report["version"] == "1.0.0"
    assert "vision" in report["frame_model_capabilities"]


def test_preflight_rejects_missing_model(monkeypatch):
    analyzer = OllamaVideoAnalyzer(frame_analysis_model="missing", summary_model="summary")
    monkeypatch.setattr(
        analyzer.session,
        "get",
        lambda endpoint, **_kwargs: DummyResponse(
            {"version": "1"} if endpoint.endswith("version") else {"models": [{"name": "summary"}]}
        ),
    )
    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        analyzer.check_environment()


def test_preflight_rejects_model_without_vision(monkeypatch):
    analyzer = OllamaVideoAnalyzer(frame_analysis_model="text", summary_model="text")
    monkeypatch.setattr(
        analyzer.session,
        "get",
        lambda endpoint, **_kwargs: DummyResponse(
            {"version": "1"} if endpoint.endswith("version") else {"models": [{"name": "text"}]}
        ),
    )
    monkeypatch.setattr(
        analyzer.session,
        "post",
        lambda *_args, **_kwargs: DummyResponse({"capabilities": ["completion"]}),
    )

    with pytest.raises(ModelCapabilityError, match="vision"):
        analyzer.check_environment()


def test_frame_request_uses_json_schema_jpeg_and_telemetry(monkeypatch):
    analyzer = OllamaVideoAnalyzer(preflight=False, request_retries=0)
    calls = []

    def fake_post(_endpoint, json=None, **_kwargs):
        calls.append(json)
        content = {
            "description": "frame",
            "objects": [],
            "actions": [],
            "visible_text": [],
            "tags": [],
        }
        return DummyResponse(
            {
                "message": {"content": json_module.dumps(content)},
                "prompt_eval_count": 4,
                "eval_count": 6,
                "total_duration": 100,
            }
        )

    json_module = json
    monkeypatch.setattr(analyzer.session, "post", fake_post)
    frame = Frame(np.zeros((2000, 1000, 3), dtype=np.uint8), 0.0, SceneType.STATIC)

    result = analyzer._analyze_frame(frame)

    assert result.description == "frame"
    assert calls[0]["format"]["required"][0] == "description"
    encoded = calls[0]["messages"][-1]["images"][0]
    assert base64.b64decode(encoded).startswith(b"\xff\xd8")
    assert analyzer._usage.total_tokens == 10


def test_custom_processor_mapping_is_normalized():
    analyzer = OllamaVideoAnalyzer(
        custom_frame_processor=lambda _frame: {"description": "custom", "objects": ["x"]},
        preflight=False,
    )
    frame = Frame(np.zeros((2, 2, 3), dtype=np.uint8), 2.0, SceneType.STATIC)

    result = analyzer._analyze_frame(frame)

    assert isinstance(result, FrameAnalysis)
    assert result.description == "custom"
    assert result.objects == ["x"]


def test_custom_processor_rejects_invalid_mapping_fields():
    analyzer = OllamaVideoAnalyzer(
        custom_frame_processor=lambda _frame: {
            "description": "custom",
            "objects": "not-an-array",
        },
        preflight=False,
    )
    frame = Frame(np.zeros((2, 2, 3), dtype=np.uint8), 2.0, SceneType.STATIC)

    with pytest.raises(ConfigurationError, match="string list"):
        analyzer._analyze_frame(frame)


def test_summary_payload_enforces_required_array_types():
    analyzer = OllamaVideoAnalyzer(preflight=False)
    with pytest.raises(ValueError, match="events"):
        analyzer._summary_from_payload(
            {"detailed": "long", "brief": "short", "events": "not-an-array"},
            "timeline",
            "transcript",
        )
