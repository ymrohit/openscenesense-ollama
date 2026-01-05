from typing import List, Dict, Tuple, Optional, Callable
import logging
from .models import (
    AnalysisPrompts,
    Frame,
    AudioSegment,
    SceneType,
    FrameAnalysis,
    SummaryResult,
    AnalysisMetadata,
    ModelsUsed,
    AnalysisResult,
)
from .frame_selectors import FrameSelector,DynamicFrameSelector
from .transcriber import AudioTranscriber
import numpy as np
from PIL import Image
import io
import base64
import requests
import time
import re
from .video_utils import get_video_duration

class OllamaVideoAnalyzer:
    def __init__(
            self,
            frame_analysis_model: str = "ministral-3:latest",
            summary_model: str = "ministral-3:latest",
            host: str = "http://localhost:11434",
            min_frames: int = 8,
            max_frames: int = 64,
            frames_per_minute: float = 4.0,
            frame_selector: Optional[FrameSelector] = None,
            audio_transcriber: Optional[AudioTranscriber] = None,
            prompts: Optional[AnalysisPrompts] = None,
            custom_frame_processor: Optional[Callable[[Frame], Dict]] = None,
            log_level: int = logging.INFO,
            request_timeout: float = 120.0,
            request_retries: int = 3,
            request_backoff: float = 1.0,
            context_max_chars: int = 0,
            audio_context_max_chars: int = 0,
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
        self.session = requests.Session()
        self.request_timeout = request_timeout
        self.request_retries = max(0, request_retries)
        self.request_backoff = max(0.0, request_backoff)
        self.retry_status_codes = {408, 429, 500, 502, 503, 504}
        self.context_max_chars = max(0, context_max_chars)
        self.audio_context_max_chars = max(0, audio_context_max_chars)
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

    def _format_frame_descriptions(self, descriptions: List[FrameAnalysis]) -> str:
        """Format frame descriptions for the summary prompt"""
        formatted = []
        for desc in descriptions:
            formatted.append(f"Time {desc.timestamp:.2f}s ({desc.scene_type}): {desc.description}")
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

    def _truncate_text(self, text: str, max_chars: int) -> str:
        compact = " ".join(text.split())
        if max_chars <= 0 or len(compact) <= max_chars:
            return compact
        clipped = compact[:max_chars].rsplit(" ", 1)[0]
        return f"{clipped}..." if clipped else compact[:max_chars]

    def _is_low_signal_audio(self, text: str) -> bool:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if len(tokens) < 8:
            return True
        unique_ratio = len(set(tokens)) / len(tokens)
        return unique_ratio < 0.2

    def _build_context_note(self, context: Optional[str], audio_text: Optional[str]) -> Optional[str]:
        parts = []
        if context:
            parts.append(f"Visual context: {self._truncate_text(context, self.context_max_chars)}")
        if audio_text:
            parts.append(f"Audio context: {self._truncate_text(audio_text, self.audio_context_max_chars)}")
        if not parts:
            return None

        header = (
            "Use the following context for continuity only. "
            "Do not reference it explicitly unless it is directly visible in the frame."
        )
        return f"{header}\n" + "\n".join(parts)

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = self.request_backoff * (2 ** (attempt - 1))
        if delay > 0:
            time.sleep(delay)

    def _post_with_retries(self, payload: Dict, request_label: str) -> requests.Response:
        attempts = self.request_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if response.status_code in self.retry_status_codes:
                    raise requests.exceptions.HTTPError(
                        f"Retryable HTTP status {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= attempts:
                    raise
                self.logger.warning(
                    "Ollama %s request failed (attempt %s/%s): %s",
                    request_label,
                    attempt,
                    attempts,
                    exc,
                )
                self._sleep_with_backoff(attempt)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status in self.retry_status_codes and attempt < attempts:
                    self.logger.warning(
                        "Ollama %s request returned %s (attempt %s/%s)",
                        request_label,
                        status,
                        attempt,
                        attempts,
                    )
                    self._sleep_with_backoff(attempt)
                    continue
                raise

    def _analyze_frame(self, frame: Frame, prompt_override: Optional[str] = None) -> FrameAnalysis:
        """Analyze a single frame using the frame analysis model"""
        if self.custom_frame_processor:
            return self.custom_frame_processor(frame)

        base64_image = self._frame_to_base64(frame.image)
        prompt = prompt_override or self.prompts.frame_analysis

        messages = []
        if self.prompts.frame_analysis_system:
            messages.append({"role": "system", "content": self.prompts.frame_analysis_system})
        messages.append({"role": "user", "content": prompt, "images": [base64_image]})

        payload = {
            "model": self.frame_analysis_model,
            "messages": messages,
            "stream": False
        }

        try:
            response = self._post_with_retries(payload, "frame analysis")
            result = response.json()
            if "message" in result:
                return FrameAnalysis(
                    timestamp=frame.timestamp,
                    description=result["message"]["content"],
                    scene_type=frame.scene_type.value,
                )
            else:
                raise Exception("Invalid response format")
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {str(e)}")
            return FrameAnalysis(
                timestamp=frame.timestamp,
                description="Error analyzing frame",
                scene_type=frame.scene_type.value,
                error=str(e),
            )

    def _generate_summary(
        self,
        frame_descriptions: List[FrameAnalysis],
        audio_segments: List[AudioSegment],
        video_duration: float,
    ) -> Tuple[SummaryResult, Optional[str]]:
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
            detailed_response = self._post_with_retries(detailed_payload, "summary (detailed)")
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
            brief_response = self._post_with_retries(brief_payload, "summary (brief)")
            brief_result = brief_response.json()

            return SummaryResult(
                detailed=(
                    detailed_result["message"]["content"]
                    if "message" in detailed_result
                    else "Unable to generate detailed summary"
                ),
                brief=(
                    brief_result["message"]["content"]
                    if "message" in brief_result
                    else "Unable to generate brief summary"
                ),
                timeline=timeline,
                transcript=transcript,
            ), None
        except Exception as e:
            self.logger.error(f"Error generating summaries: {str(e)}")
            return SummaryResult(
                detailed="Error generating video summary",
                brief="Error generating brief summary",
                timeline=timeline,
                transcript=transcript,
            ), f"Summary generation failed: {str(e)}"

    def analyze_video_structured(self, video_path: str) -> AnalysisResult:
        """Provide a comprehensive video analysis using both visual and audio content"""
        self.logger.info(f"Starting video analysis for: {video_path}")

        try:
            # Extract key frames
            self.logger.info("Selecting key frames")
            frames = self.frame_selector.select_frames(video_path, self)
            self.logger.info(f"Selected {len(frames)} frames for analysis")

            # Transcribe audio if available
            audio_segments: List[AudioSegment] = []
            warnings: List[str] = []
            if self.audio_transcriber:
                self.logger.info("Starting audio transcription")
                try:
                    audio_segments = self.audio_transcriber.transcribe(video_path)
                    self.logger.info(f"Transcribed {len(audio_segments)} audio segments")
                except Exception as e:
                    warning = f"Audio transcription failed: {str(e)}"
                    warnings.append(warning)
                    self.logger.error(warning)

            # Analyze frames with context
            frame_descriptions: List[FrameAnalysis] = []
            context = None

            for i, frame in enumerate(frames):
                self.logger.info(f"Analyzing frame {i + 1}/{len(frames)} at {frame.timestamp:.2f}s")

                # Find relevant audio segments for this frame
                relevant_audio = [
                    segment for segment in audio_segments
                    if segment.start_time <= frame.timestamp <= segment.end_time
                ]

                frame_context = context
                audio_context = None
                if relevant_audio:
                    audio_text = relevant_audio[0].text
                    if audio_text and not self._is_low_signal_audio(audio_text):
                        audio_context = audio_text

                context_note = self._build_context_note(frame_context, audio_context)
                frame_prompt = self.prompts.frame_analysis
                if context_note:
                    frame_prompt = f"{frame_prompt}\n\n{context_note}"

                analysis = self._analyze_frame(frame, frame_prompt)
                frame_descriptions.append(analysis)
                context = analysis.description
                time.sleep(0.1)  # Rate limiting

            # Generate summaries with both video and audio
            fallback_duration = frames[-1].timestamp if frames else 0.0
            video_duration = get_video_duration(video_path, fallback_duration=fallback_duration)
            self.logger.info("Generating video and audio summaries")
            summaries, summary_warning = self._generate_summary(
                frame_descriptions,
                audio_segments,
                video_duration,
            )
            if summary_warning:
                warnings.append(summary_warning)

            # Collect metadata
            scene_distribution = {
                scene_type.value: len([f for f in frames if f.scene_type == scene_type])
                for scene_type in SceneType
            }

            frame_errors = [analysis for analysis in frame_descriptions if analysis.error]
            if frame_errors:
                warnings.append(f"{len(frame_errors)} frame analyses failed.")

            metadata = AnalysisMetadata(
                num_frames_analyzed=len(frames),
                num_audio_segments=len(audio_segments),
                video_duration=video_duration,
                scene_distribution=scene_distribution,
                models_used=ModelsUsed(
                    frame_analysis=self.frame_analysis_model,
                    summary=self.summary_model,
                    audio=self.audio_transcriber.__class__.__name__ if self.audio_transcriber else None,
                ),
            )
            result = AnalysisResult(
                summary=summaries,
                frame_analyses=frame_descriptions,
                audio_segments=audio_segments,
                metadata=metadata,
                warnings=warnings,
            )

            self.logger.info("Video and audio analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            raise

    def analyze_video(self, video_path: str) -> Dict:
        """Provide a comprehensive video analysis as a legacy dictionary."""
        return self.analyze_video_structured(video_path).to_legacy_dict()
