from abc import ABC, abstractmethod
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import logging
from .models import AudioSegment
import torch

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
            model_name: str = "openai/whisper-tiny",
            device: Optional[str] = None
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Initializing Whisper transcriber with model {model_name} on {device}")
        self.device = device

        # Initialize processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)

        # Disable forced decoder ids
        self.model.config.forced_decoder_ids = None

        # Target sampling rate for Whisper
        self.target_sampling_rate = 16000

    def _extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video file and return as numpy array with sampling rate using librosa"""
        try:
            # Load audio using librosa
            raw_audio, original_sampling_rate = librosa.load(
                video_path,
                sr=self.target_sampling_rate,
                mono=True
            )

            # Ensure float32 dtype and normalize
            raw_audio = raw_audio.astype(np.float32)
            if np.abs(raw_audio).max() > 1.0:
                raw_audio = raw_audio / np.abs(raw_audio).max()

            self.logger.debug(f"Raw audio shape: {raw_audio.shape}, dtype: {raw_audio.dtype}")

            return raw_audio, original_sampling_rate

        except Exception as e:
            self.logger.error(f"Error extracting audio with librosa: {str(e)}")
            raise


    def _segment_audio(self, audio: np.ndarray, sampling_rate: int, segment_duration: int = 30) -> List[
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
            audio_segments = self._segment_audio(audio_array, self.target_sampling_rate)

            transcribed_segments = []

            # Process each audio segment
            for segment_audio, start_time in audio_segments:
                # Prepare features
                input_features = self.processor(
                    [segment_audio],  # Changed here
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt"
                ).input_features.to(self.device)

                # Generate transcription
                predicted_ids = self.model.generate(input_features)

                # Decode transcription
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]  # Take first element as we process one segment at a time

                # Calculate segment duration
                duration = len(segment_audio) / self.target_sampling_rate

                # Create AudioSegment
                segment = AudioSegment(
                    text=transcription.strip(),
                    start_time=start_time,
                    end_time=start_time + duration,
                    confidence=1.0  # Note: Basic Whisper doesn't provide confidence scores
                )

                transcribed_segments.append(segment)

            self.logger.info(f"Transcription completed: {len(transcribed_segments)} segments")
            return transcribed_segments

        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise