from typing import List, Dict, Tuple, Optional, Callable
import logging
from .models import AnalysisPrompts,Frame,AudioSegment,SceneType
from .frame_selectors import FrameSelector,DynamicFrameSelector
from .transcriber import AudioTranscriber
import numpy as np
from PIL import Image
import io
import base64
import requests
import time

class OllamaVideoAnalyzer:
    def __init__(
            self,
            frame_analysis_model: str = "minicpm-v",
            summary_model: str = "llama3.2",
            host: str = "http://localhost:11434",
            min_frames: int = 8,
            max_frames: int = 64,
            frames_per_minute: float = 4.0,
            frame_selector: Optional[FrameSelector] = None,
            audio_transcriber: Optional[AudioTranscriber] = None,
            prompts: Optional[AnalysisPrompts] = None,
            custom_frame_processor: Optional[Callable[[Frame], Dict]] = None,
            log_level: int = logging.INFO
    ):
        self.frame_analysis_model = frame_analysis_model
        self.summary_model = summary_model
        self.host = host
        self.api_endpoint = f"{host}/api/chat"
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_minute = frames_per_minute
        self.frame_selector = frame_selector or DynamicFrameSelector()
        self.audio_transcriber = audio_transcriber
        self.prompts = prompts or AnalysisPrompts()
        self.custom_frame_processor = custom_frame_processor
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert a frame to base64 string"""
        image = Image.fromarray(frame)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _format_transcript(self, segments: List[AudioSegment]) -> str:
        """Format audio transcript with timestamps"""
        formatted = []
        for segment in segments:
            formatted.append(
                f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s]: {segment.text}"
            )
        return "\n".join(formatted)

    def _format_frame_descriptions(self, descriptions: List[Dict]) -> str:
        """Format frame descriptions for the summary prompt"""
        formatted = []
        for desc in descriptions:
            formatted.append(f"Time {desc['timestamp']:.2f}s ({desc['scene_type']}): {desc['description']}")
        return "\n".join(formatted)

    def _calculate_dynamic_frame_count(self, video_duration: float, scene_changes: List[float]) -> int:
        """Calculate optimal number of frames to analyze"""
        base_frames = int(video_duration / 60 * self.frames_per_minute)
        scene_density = len(scene_changes) / video_duration if video_duration > 0 else 0
        scene_multiplier = min(2.0, max(0.5, scene_density * 30))
        optimal_frames = int(base_frames * scene_multiplier)
        return min(self.max_frames, max(self.min_frames, optimal_frames))

    def _calculate_uniform_frame_count(self, video_duration: float) -> int:
        """Calculate the number of frames to select uniformly based on video duration."""
        base_frames = int((video_duration / 60) * self.frames_per_minute)
        optimal_frames = min(self.max_frames, max(self.min_frames, base_frames))
        self.logger.debug(
            f"Calculated uniform frame count: {optimal_frames} (Base: {base_frames}, Min: {self.min_frames}, Max: {self.max_frames})")
        return optimal_frames

    def _analyze_frame(self, frame: Frame, context: Optional[str] = None) -> Dict:
        """Analyze a single frame using the frame analysis model"""
        if self.custom_frame_processor:
            return self.custom_frame_processor(frame)

        base64_image = self._frame_to_base64(frame.image)
        prompt = self.prompts.frame_analysis
        if context:
            prompt += f"\nPrevious context: {context}"

        payload = {
            "model": self.frame_analysis_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False
        }

        try:
            response = requests.post(self.api_endpoint, json=payload)
            result = response.json()
            if "message" in result:
                return {
                    "timestamp": frame.timestamp,
                    "description": result["message"]["content"],
                    "scene_type": frame.scene_type.value
                }
            else:
                raise Exception("Invalid response format")
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {str(e)}")
            return {
                "timestamp": frame.timestamp,
                "description": "Error analyzing frame",
                "scene_type": frame.scene_type.value
            }

    def _generate_summary(self, frame_descriptions: List[Dict], audio_segments: List[AudioSegment],
                          video_duration: float) -> Dict:
        """Generate summaries using both video and audio information"""
        timeline = self._format_frame_descriptions(frame_descriptions)
        transcript = self._format_transcript(audio_segments) if audio_segments else "No audio transcript available."

        try:
            # Get detailed summary
            detailed_prompt = self.prompts.detailed_summary.format(
                duration=video_duration,
                timeline=timeline,
                transcript=transcript
            )
            detailed_payload = {
                "model": self.summary_model,
                "messages": [{"role": "user", "content": detailed_prompt}],
                "stream": False
            }
            detailed_response = requests.post(self.api_endpoint, json=detailed_payload)
            detailed_result = detailed_response.json()

            # Get brief summary
            brief_prompt = self.prompts.brief_summary.format(
                duration=video_duration,
                timeline=timeline,
                transcript=transcript
            )
            brief_payload = {
                "model": self.summary_model,
                "messages": [{"role": "user", "content": brief_prompt}],
                "stream": False
            }
            brief_response = requests.post(self.api_endpoint, json=brief_payload)
            brief_result = brief_response.json()

            return {
                "detailed": detailed_result["message"]["content"] if "message" in detailed_result else "Unable to generate detailed summary",
                "brief": brief_result["message"]["content"] if "message" in brief_result else "Unable to generate brief summary",
                "timeline": timeline,
                "transcript": transcript
            }
        except Exception as e:
            self.logger.error(f"Error generating summaries: {str(e)}")
            return {
                "detailed": "Error generating video summary",
                "brief": "Error generating brief summary",
                "timeline": timeline,
                "transcript": transcript
            }

    def analyze_video(self, video_path: str) -> Dict:
        """Provide a comprehensive video analysis using both visual and audio content"""
        self.logger.info(f"Starting video analysis for: {video_path}")

        try:
            # Extract key frames
            self.logger.info("Selecting key frames")
            frames = self.frame_selector.select_frames(video_path, self)
            self.logger.info(f"Selected {len(frames)} frames for analysis")

            # Transcribe audio if available
            audio_segments = []
            if self.audio_transcriber:
                self.logger.info("Starting audio transcription")
                try:
                    audio_segments = self.audio_transcriber.transcribe(video_path)
                    self.logger.info(f"Transcribed {len(audio_segments)} audio segments")
                except Exception as e:
                    self.logger.error(f"Audio transcription failed: {str(e)}")

            # Analyze frames with context
            frame_descriptions = []
            context = None

            for i, frame in enumerate(frames):
                self.logger.info(f"Analyzing frame {i + 1}/{len(frames)} at {frame.timestamp:.2f}s")

                # Find relevant audio segments for this frame
                relevant_audio = [
                    segment for segment in audio_segments
                    if segment.start_time <= frame.timestamp <= segment.end_time
                ]

                # Add audio context to frame analysis if available
                frame_context = f"{context}\n\nAudio context: {relevant_audio[0].text if relevant_audio else 'No audio'}" if context else None

                analysis = self._analyze_frame(frame, frame_context)
                frame_descriptions.append(analysis)
                context = analysis["description"]
                time.sleep(0.1)  # Rate limiting

            # Generate summaries with both video and audio
            video_duration = frames[-1].timestamp if frames else 0
            self.logger.info("Generating video and audio summaries")
            summaries = self._generate_summary(frame_descriptions, audio_segments, video_duration)

            # Collect metadata
            scene_distribution = {
                scene_type.value: len([f for f in frames if f.scene_type == scene_type])
                for scene_type in SceneType
            }

            result = {
                "summary": summaries["detailed"],
                "brief_summary": summaries["brief"],
                "timeline": summaries["timeline"],
                "transcript": summaries["transcript"],
                "frame_analyses": frame_descriptions,
                "audio_segments": [
                    {
                        "text": segment.text,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "confidence": segment.confidence
                    }
                    for segment in audio_segments
                ],
                "metadata": {
                    "num_frames_analyzed": len(frames),
                    "num_audio_segments": len(audio_segments),
                    "video_duration": video_duration,
                    "scene_distribution": scene_distribution,
                    "models_used": {
                        "frame_analysis": self.frame_analysis_model,
                        "summary": self.summary_model,
                        "audio": self.audio_transcriber.__class__.__name__ if self.audio_transcriber else None
                    }
                }
            }

            self.logger.info("Video and audio analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            raise
