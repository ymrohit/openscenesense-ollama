from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.2"


class SceneType(Enum):
    STATIC = "static"
    ACTION = "action"
    TRANSITION = "transition"


class SelectionReason(Enum):
    OPENING = "opening"
    CLOSING = "closing"
    SCENE_CHANGE = "scene_change"
    UNIFORM_FILL = "uniform_fill"


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    scene_type: SceneType
    difference_score: float = 0.0
    selection_reason: str = SelectionReason.UNIFORM_FILL.value


@dataclass
class AudioSegment:
    text: str
    start_time: float
    end_time: float
    confidence: float = 0.0


@dataclass
class AnalysisPrompts:
    frame_analysis: str = (
        "Describe what is visible in this video frame. Identify important objects, actions, "
        "changes, and visible text. Do not speculate beyond the image."
    )
    frame_analysis_system: str = (
        "You are a visual analyst. Describe only what is visible in the frame. "
        "Use supplied context only for continuity and never mention it explicitly."
    )
    detailed_summary: str = (
        "Create a comprehensive narrative integrating the visual timeline and audio transcript "
        "from this {duration:.1f}-second video.\n\nTimeline:\n{timeline}\n\n"
        "Audio transcript:\n{transcript}"
    )
    brief_summary: str = (
        "Create a concise two-to-three sentence summary of this {duration:.1f}-second video.\n"
        "Timeline:\n{timeline}\n\nAudio transcript:\n{transcript}"
    )


@dataclass
class TimelineEvent:
    start_time: float
    end_time: float
    description: str
    source_frame_timestamps: list[float]
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    visible_text: list[str] = field(default_factory=list)


@dataclass
class FrameAnalysis:
    timestamp: float
    description: str
    scene_type: str
    selection_reason: str = SelectionReason.UNIFORM_FILL.value
    difference_score: float = 0.0
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    visible_text: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class SummaryResult:
    detailed: str
    brief: str
    timeline: str = ""
    transcript: str = ""


@dataclass
class VideoMetadata:
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0


@dataclass
class SelectionMetadata:
    strategy: str = "unknown"
    selected_frame_count: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsUsed:
    frame_analysis: str
    summary: str
    audio: str | None
    provider: str = "unknown"


@dataclass
class PerformanceMetadata:
    total_seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class UsageMetadata:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisMetadata:
    num_frames_analyzed: int
    num_audio_segments: int
    video_duration: float
    scene_distribution: dict[str, int]
    models_used: ModelsUsed
    video: VideoMetadata | None = None
    selection: SelectionMetadata | None = None
    performance: PerformanceMetadata = field(default_factory=PerformanceMetadata)
    usage: UsageMetadata = field(default_factory=UsageMetadata)
    successful_frame_analyses: int = 0
    failed_frame_analyses: int = 0

    def structured(self) -> dict[str, Any]:
        video = self.video or VideoMetadata(duration=self.video_duration)
        selection = self.selection or SelectionMetadata(
            selected_frame_count=self.num_frames_analyzed
        )
        return {
            "num_frames_analyzed": self.num_frames_analyzed,
            "num_audio_segments": self.num_audio_segments,
            "video_duration": self.video_duration,
            "scene_distribution": dict(self.scene_distribution),
            "models_used": asdict(self.models_used),
            "video": asdict(video),
            "selection": asdict(selection),
            "models": {
                "provider": self.models_used.provider,
                "vision": self.models_used.frame_analysis,
                "summary": self.models_used.summary,
                "audio": self.models_used.audio,
            },
            "performance": asdict(self.performance),
            "usage": asdict(self.usage),
            "successful_frame_analyses": self.successful_frame_analyses,
            "failed_frame_analyses": self.failed_frame_analyses,
        }


@dataclass
class AnalysisResult:
    summary: SummaryResult
    frame_analyses: list[FrameAnalysis]
    audio_segments: list[AudioSegment]
    metadata: AnalysisMetadata
    timeline: list[TimelineEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": {"detailed": self.summary.detailed, "brief": self.summary.brief},
            "timeline": [asdict(event) for event in self.timeline],
            "frame_analyses": [asdict(item) for item in self.frame_analyses],
            "audio_segments": [asdict(item) for item in self.audio_segments],
            "metadata": self.metadata.structured(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        timeline_text = self.summary.timeline or "\n".join(
            f"Time {event.start_time:.2f}s: {event.description}" for event in self.timeline
        )
        transcript = self.summary.transcript or "\n".join(
            f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s]: {segment.text}"
            for segment in self.audio_segments
        )
        legacy_frames = []
        for analysis in self.frame_analyses:
            item = asdict(analysis)
            if not analysis.error:
                item.pop("error", None)
            legacy_frames.append(item)
        legacy_metadata = {
            "num_frames_analyzed": self.metadata.num_frames_analyzed,
            "num_audio_segments": self.metadata.num_audio_segments,
            "video_duration": self.metadata.video_duration,
            "scene_distribution": dict(self.metadata.scene_distribution),
            "models_used": asdict(self.metadata.models_used),
            **self.metadata.structured(),
        }
        return {
            "summary": self.summary.detailed,
            "brief_summary": self.summary.brief,
            "timeline": timeline_text,
            "transcript": transcript,
            "frame_analyses": legacy_frames,
            "audio_segments": [asdict(item) for item in self.audio_segments],
            "metadata": legacy_metadata,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @staticmethod
    def schema() -> dict[str, Any]:
        return analysis_result_schema()


def analysis_result_schema() -> dict[str, Any]:
    return copy.deepcopy(ANALYSIS_RESULT_SCHEMA)


ANALYSIS_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://openscenesense.dev/schemas/analysis-result-1.2.json",
    "title": "AnalysisResult",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "summary",
        "timeline",
        "frame_analyses",
        "audio_segments",
        "metadata",
        "warnings",
        "errors",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["detailed", "brief"],
            "properties": {"detailed": {"type": "string"}, "brief": {"type": "string"}},
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "start_time",
                    "end_time",
                    "description",
                    "source_frame_timestamps",
                    "objects",
                    "actions",
                    "visible_text",
                ],
                "properties": {
                    "start_time": {"type": "number", "minimum": 0},
                    "end_time": {"type": "number", "minimum": 0},
                    "description": {"type": "string"},
                    "source_frame_timestamps": {"type": "array", "items": {"type": "number"}},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "frame_analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "timestamp",
                    "description",
                    "scene_type",
                    "selection_reason",
                    "difference_score",
                    "objects",
                    "actions",
                    "visible_text",
                    "tags",
                    "error",
                ],
                "properties": {
                    "timestamp": {"type": "number", "minimum": 0},
                    "description": {"type": "string"},
                    "scene_type": {"type": "string"},
                    "selection_reason": {
                        "enum": ["opening", "closing", "scene_change", "uniform_fill"]
                    },
                    "difference_score": {"type": "number"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "error": {"type": ["string", "null"]},
                },
            },
        },
        "audio_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "start_time", "end_time", "confidence"],
                "properties": {
                    "text": {"type": "string"},
                    "start_time": {"type": "number", "minimum": 0},
                    "end_time": {"type": "number", "minimum": 0},
                    "confidence": {"type": "number"},
                },
            },
        },
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "num_frames_analyzed",
                "num_audio_segments",
                "video_duration",
                "scene_distribution",
                "models_used",
                "video",
                "selection",
                "models",
                "performance",
                "usage",
                "successful_frame_analyses",
                "failed_frame_analyses",
            ],
            "properties": {
                "num_frames_analyzed": {"type": "integer", "minimum": 0},
                "num_audio_segments": {"type": "integer", "minimum": 0},
                "video_duration": {"type": "number", "minimum": 0},
                "scene_distribution": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
                "models_used": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["frame_analysis", "summary", "audio", "provider"],
                    "properties": {
                        "frame_analysis": {"type": "string"},
                        "summary": {"type": "string"},
                        "audio": {"type": ["string", "null"]},
                        "provider": {"type": "string"},
                    },
                },
                "video": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["duration", "fps", "frame_count", "width", "height"],
                    "properties": {
                        "duration": {"type": "number", "minimum": 0},
                        "fps": {"type": "number", "minimum": 0},
                        "frame_count": {"type": "integer", "minimum": 0},
                        "width": {"type": "integer", "minimum": 0},
                        "height": {"type": "integer", "minimum": 0},
                    },
                },
                "selection": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy", "selected_frame_count", "parameters"],
                    "properties": {
                        "strategy": {"type": "string"},
                        "selected_frame_count": {"type": "integer", "minimum": 0},
                        "parameters": {"type": "object"},
                    },
                },
                "models": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["provider", "vision", "summary", "audio"],
                    "properties": {
                        "provider": {"type": "string"},
                        "vision": {"type": "string"},
                        "summary": {"type": "string"},
                        "audio": {"type": ["string", "null"]},
                    },
                },
                "performance": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["total_seconds", "stage_seconds"],
                    "properties": {
                        "total_seconds": {"type": "number", "minimum": 0},
                        "stage_seconds": {
                            "type": "object",
                            "additionalProperties": {"type": "number", "minimum": 0},
                        },
                    },
                },
                "usage": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "provider_details",
                    ],
                    "properties": {
                        "input_tokens": {"type": "integer", "minimum": 0},
                        "output_tokens": {"type": "integer", "minimum": 0},
                        "total_tokens": {"type": "integer", "minimum": 0},
                        "provider_details": {"type": "object"},
                    },
                },
                "successful_frame_analyses": {"type": "integer", "minimum": 0},
                "failed_frame_analyses": {"type": "integer", "minimum": 0},
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "errors": {"type": "array", "items": {"type": "string"}},
    },
}
