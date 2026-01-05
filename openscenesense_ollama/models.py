
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from typing import List, Dict, Optional, Any
import copy
class SceneType(Enum):
    STATIC = "static"
    ACTION = "action"
    TRANSITION = "transition"


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    scene_type: SceneType
    difference_score: float = 0.0


@dataclass
class AudioSegment:
    """Represents a transcribed segment of audio with timing information"""
    text: str
    start_time: float
    end_time: float
    confidence: float = 0.0

@dataclass
class AnalysisPrompts:
    """Customizable prompts for video and audio analysis"""
    frame_analysis: str = "Describe what's happening in this moment of the video, focusing on important actions, objects, or changes."
    frame_analysis_system: str = (
        "You are a visual analyst. Describe only what is visible in the frame. "
        "Do not mention prior responses, do not apologize, and do not speculate. "
        "If context is provided, use it only for continuity without referencing it directly."
    )

    detailed_summary: str = """You are an expert video and audio analyst and storyteller. Based on the following chronological descriptions 
    of key moments from a {duration:.1f}-second video, along with its audio transcript, create a comprehensive narrative.

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Please provide a detailed summary that:
    1. Tells a cohesive story integrating both visual and audio elements
    2. Captures the progression and flow of events
    3. Highlights significant moments, changes, and patterns
    4. Notes any important dialogue or audio cues
    5. Identifies relationships between what is seen and heard
    6. Maintains a natural, engaging narrative style

    Focus on creating a flowing narrative that combines visual and audio elements."""

    brief_summary: str = """You are an expert video and audio analyst. Based on the following information from a {duration:.1f}-second video:

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Provide a concise 2-3 line summary that captures the essence of both the visual and audio content."""


@dataclass
class FrameAnalysis:
    timestamp: float
    description: str
    scene_type: str
    error: Optional[str] = None


@dataclass
class SummaryResult:
    detailed: str
    brief: str
    timeline: str
    transcript: str


@dataclass
class ModelsUsed:
    frame_analysis: str
    summary: str
    audio: Optional[str]


@dataclass
class AnalysisMetadata:
    num_frames_analyzed: int
    num_audio_segments: int
    video_duration: float
    scene_distribution: Dict[str, int]
    models_used: ModelsUsed


@dataclass
class AnalysisResult:
    summary: SummaryResult
    frame_analyses: List[FrameAnalysis]
    audio_segments: List[AudioSegment]
    metadata: AnalysisMetadata
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_legacy_dict(self) -> Dict[str, Any]:
        frame_analyses = []
        for analysis in self.frame_analyses:
            item = {
                "timestamp": analysis.timestamp,
                "description": analysis.description,
                "scene_type": analysis.scene_type,
            }
            if analysis.error:
                item["error"] = analysis.error
            frame_analyses.append(item)

        return {
            "summary": self.summary.detailed,
            "brief_summary": self.summary.brief,
            "timeline": self.summary.timeline,
            "transcript": self.summary.transcript,
            "frame_analyses": frame_analyses,
            "audio_segments": [asdict(segment) for segment in self.audio_segments],
            "metadata": asdict(self.metadata),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def schema() -> Dict[str, Any]:
        return analysis_result_schema()


def analysis_result_schema() -> Dict[str, Any]:
    return copy.deepcopy(ANALYSIS_RESULT_SCHEMA)


ANALYSIS_RESULT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AnalysisResult",
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "frame_analyses", "audio_segments", "metadata", "warnings"],
    "properties": {
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["detailed", "brief", "timeline", "transcript"],
            "properties": {
                "detailed": {"type": "string"},
                "brief": {"type": "string"},
                "timeline": {"type": "string"},
                "transcript": {"type": "string"},
            },
        },
        "frame_analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["timestamp", "description", "scene_type"],
                "properties": {
                    "timestamp": {"type": "number"},
                    "description": {"type": "string"},
                    "scene_type": {"type": "string"},
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
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
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
            ],
            "properties": {
                "num_frames_analyzed": {"type": "integer"},
                "num_audio_segments": {"type": "integer"},
                "video_duration": {"type": "number"},
                "scene_distribution": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
                "models_used": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["frame_analysis", "summary", "audio"],
                    "properties": {
                        "frame_analysis": {"type": "string"},
                        "summary": {"type": "string"},
                        "audio": {"type": ["string", "null"]},
                    },
                },
            },
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}
