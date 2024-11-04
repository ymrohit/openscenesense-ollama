
from dataclasses import dataclass
from enum import Enum
import numpy as np
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
