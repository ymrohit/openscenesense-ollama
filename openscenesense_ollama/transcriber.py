from abc import ABC, abstractmethod
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import logging
import torch
import ffmpeg
import re
from .models import AudioSegment
from .exceptions import TranscriptionError


def _collapse_repeated_phrases(
    text: str,
    min_repeats: int = 5,
    max_phrase_words: int = 5,
) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if len(parts) < min_repeats:
        return text

    result = []
    i = 0
    while i < len(parts):
        phrase = parts[i]
        normalized = " ".join(phrase.lower().split())
        j = i + 1
        while j < len(parts) and " ".join(parts[j].lower().split()) == normalized:
            j += 1
        count = j - i
        word_count = len(re.findall(r"\w+", phrase))
        if count >= min_repeats and word_count <= max_phrase_words:
            result.append(f"{phrase} (repeated {count}x)")
        else:
            result.extend(parts[i:j])
        i = j

    return " ".join(result)

class AudioTranscriber(ABC):
    """Abstract base class for audio transcription strategies"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def transcribe(self, video_path: str) -> List[AudioSegment]:
        pass


class WhisperTranscriber(AudioTranscriber):
    """Audio transcription using OpenAI's Whisper model with direct model interaction"""

    def __init__(
            self,
            model_name: str = "openai/whisper-small",
            device: Optional[str] = None,
            language: Optional[str] = None,
            task: str = "transcribe",
            collapse_repetitions: bool = False,
            min_repeated_phrases: int = 5,
            max_repeat_phrase_words: int = 5,
            segment_duration: int = 30,
            beam_size: int = 1,
            temperature: Optional[float] = None,
            condition_on_prev_tokens: Optional[bool] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Initializing Whisper transcriber with model {model_name} on {device}")
        self.device = device
        self.language = language
        self.task = task
        self.collapse_repetitions = collapse_repetitions
        self.min_repeated_phrases = max(1, min_repeated_phrases)
        self.max_repeat_phrase_words = max(1, max_repeat_phrase_words)
        self.segment_duration = max(1, segment_duration)
        self.beam_size = max(1, beam_size)
        self.temperature = temperature
        self.condition_on_prev_tokens = condition_on_prev_tokens

        # Initialize processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Disable forced decoder ids
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.language = language
        self.model.generation_config.task = task

        # Target sampling rate for Whisper
        self.target_sampling_rate = 16000

    def _has_audio_stream(self, video_path: str) -> bool:
        try:
            info = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            message = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            raise TranscriptionError(f"ffprobe failed: {message}") from e

        streams = info.get("streams", [])
        return any(stream.get("codec_type") == "audio" for stream in streams)

    def _extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video file and return as numpy array with sampling rate using ffmpeg"""
        try:
            if not self._has_audio_stream(video_path):
                raise TranscriptionError(f"No audio stream found in {video_path}")

            out, _ = (
                ffmpeg.input(video_path)
                .output(
                    "pipe:",
                    format="f32le",
                    ac=1,
                    ar=self.target_sampling_rate,
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
            raw_audio = np.frombuffer(out, np.float32)
            if raw_audio.size == 0:
                raise TranscriptionError(f"No audio samples extracted from {video_path}")

            if np.abs(raw_audio).max() > 1.0:
                raw_audio = raw_audio / np.abs(raw_audio).max()

            self.logger.debug("Raw audio shape: %s, dtype: %s", raw_audio.shape, raw_audio.dtype)
            return raw_audio, self.target_sampling_rate
        except Exception as e:
            if isinstance(e, TranscriptionError):
                self.logger.error(str(e))
                raise

            self.logger.error(f"Error extracting audio with ffmpeg: {str(e)}")
            raise TranscriptionError(f"Error extracting audio with ffmpeg: {str(e)}") from e


    def _segment_audio(self, audio: np.ndarray, sampling_rate: int, segment_duration: int) -> List[
        Tuple[np.ndarray, float]]:
        """Segment audio into chunks for processing"""
        segment_length = segment_duration * sampling_rate
        segments = []
        start_idx = 0

        while start_idx < len(audio):
            end_idx = min(start_idx + segment_length, len(audio))
            segment = audio[start_idx:end_idx]
            start_time = start_idx / sampling_rate
            segments.append((segment, start_time))
            start_idx = end_idx

        return segments

    def transcribe(self, video_path: str) -> List[AudioSegment]:
        """Transcribe audio from video file using Whisper"""
        self.logger.info(f"Starting audio transcription for {video_path}")

        try:
            # Extract audio
            audio_array, original_sampling_rate = self._extract_audio(video_path)

            # Segment audio into manageable chunks
            audio_segments = self._segment_audio(
                audio_array,
                self.target_sampling_rate,
                self.segment_duration,
            )

            transcribed_segments = []

            # Process each audio segment
            for segment_audio, start_time in audio_segments:
                # Prepare features
                features = self.processor(
                    [segment_audio],
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = features.input_features.to(self.device)
                attention_mask = features.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Generate transcription
                generate_kwargs = {}
                if self.task:
                    generate_kwargs["task"] = self.task
                if self.language:
                    generate_kwargs["language"] = self.language
                if attention_mask is not None:
                    generate_kwargs["attention_mask"] = attention_mask
                if self.beam_size > 1:
                    generate_kwargs["num_beams"] = self.beam_size
                    generate_kwargs["do_sample"] = False
                if self.temperature is not None:
                    generate_kwargs["temperature"] = self.temperature
                if self.condition_on_prev_tokens is not None:
                    generate_kwargs["condition_on_prev_tokens"] = self.condition_on_prev_tokens

                with torch.inference_mode():
                    predicted_ids = self.model.generate(input_features, **generate_kwargs)

                # Decode transcription
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]  # Take first element as we process one segment at a time
                transcription = transcription.strip()
                if self.collapse_repetitions:
                    transcription = _collapse_repeated_phrases(
                        transcription,
                        min_repeats=self.min_repeated_phrases,
                        max_phrase_words=self.max_repeat_phrase_words,
                    )

                # Calculate segment duration
                duration = len(segment_audio) / self.target_sampling_rate

                # Create AudioSegment
                segment = AudioSegment(
                    text=transcription,
                    start_time=start_time,
                    end_time=start_time + duration,
                    confidence=1.0  # Note: Basic Whisper doesn't provide confidence scores
                )

                transcribed_segments.append(segment)

            self.logger.info(f"Transcription completed: {len(transcribed_segments)} segments")
            return transcribed_segments

        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionError(f"Transcription failed: {str(e)}") from e
