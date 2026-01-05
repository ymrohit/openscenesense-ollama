from openscenesense_ollama.models import (
    AnalysisResult,
    SummaryResult,
    FrameAnalysis,
    AudioSegment,
    AnalysisMetadata,
    ModelsUsed,
    analysis_result_schema,
)


def _build_result() -> AnalysisResult:
    summary = SummaryResult(
        detailed="detailed summary",
        brief="brief summary",
        timeline="timeline",
        transcript="transcript",
    )
    frames = [
        FrameAnalysis(timestamp=1.0, description="ok", scene_type="static"),
        FrameAnalysis(
            timestamp=2.0,
            description="Error analyzing frame",
            scene_type="transition",
            error="timeout",
        ),
    ]
    audio_segments = [AudioSegment(text="hello", start_time=0.0, end_time=1.2, confidence=0.9)]
    metadata = AnalysisMetadata(
        num_frames_analyzed=2,
        num_audio_segments=1,
        video_duration=10.0,
        scene_distribution={"static": 1, "transition": 1, "action": 0},
        models_used=ModelsUsed(frame_analysis="vision", summary="summary", audio="audio"),
    )
    return AnalysisResult(
        summary=summary,
        frame_analyses=frames,
        audio_segments=audio_segments,
        metadata=metadata,
        warnings=["warning"],
    )


def test_analysis_result_to_dict():
    result = _build_result()
    structured = result.to_dict()

    assert structured["summary"]["brief"] == "brief summary"
    assert structured["frame_analyses"][1]["error"] == "timeout"
    assert structured["metadata"]["models_used"]["frame_analysis"] == "vision"


def test_analysis_result_to_legacy_dict():
    result = _build_result()
    legacy = result.to_legacy_dict()

    assert legacy["summary"] == "detailed summary"
    assert legacy["brief_summary"] == "brief summary"
    assert legacy["warnings"] == ["warning"]
    assert "error" not in legacy["frame_analyses"][0]
    assert legacy["frame_analyses"][1]["error"] == "timeout"


def test_analysis_result_schema_structure():
    schema = analysis_result_schema()
    assert schema["title"] == "AnalysisResult"
    assert "summary" in schema["properties"]
