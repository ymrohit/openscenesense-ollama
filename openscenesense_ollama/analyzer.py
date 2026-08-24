from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Literal

import requests
from PIL import Image

from .cache import AnalysisCache, analysis_cache_key
from .exceptions import (
    ConfigurationError,
    ModelCapabilityError,
    ModelNotFoundError,
    ProviderConnectionError,
    ResponseValidationError,
    VideoLoadError,
)
from .frame_selectors import DynamicFrameSelector, FrameSelector
from .models import (
    AnalysisMetadata,
    AnalysisPrompts,
    AnalysisResult,
    AudioSegment,
    Frame,
    FrameAnalysis,
    ModelsUsed,
    PerformanceMetadata,
    SceneType,
    SelectionMetadata,
    SummaryResult,
    TimelineEvent,
    UsageMetadata,
)
from .progress import ProgressEvent
from .transcriber import AudioTranscriber
from .video_utils import probe_video

ProgressCallback = Callable[[ProgressEvent], None]


FRAME_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["description", "objects", "actions", "visible_text", "tags"],
    "properties": {
        "description": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}

SUMMARY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["detailed", "brief", "events"],
    "properties": {
        "detailed": {"type": "string"},
        "brief": {"type": "string"},
        "events": {
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
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "description": {"type": "string"},
                    "source_frame_timestamps": {"type": "array", "items": {"type": "number"}},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama response must be a JSON object.")
    return parsed


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return value


def _optional_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigurationError(f"custom_frame_processor field {key!r} must be a string list.")
    return value


class OllamaVideoAnalyzer:
    def __init__(
        self,
        frame_analysis_model: str = "ministral-3:latest",
        summary_model: str = "ministral-3:latest",
        host: str = "http://localhost:11434",
        min_frames: int = 8,
        max_frames: int = 64,
        frames_per_minute: float = 4.0,
        frame_selector: FrameSelector | None = None,
        audio_transcriber: AudioTranscriber | None = None,
        prompts: AnalysisPrompts | None = None,
        custom_frame_processor: Callable[[Frame], FrameAnalysis | Mapping[str, Any]] | None = None,
        log_level: int = logging.INFO,
        request_timeout: float = 120.0,
        request_retries: int = 3,
        request_backoff: float = 1.0,
        context_max_chars: int = 1000,
        audio_context_max_chars: int = 1000,
        *,
        context_mode: Literal["sliding", "independent"] = "sliding",
        max_workers: int = 1,
        structured_output: bool = True,
        temperature: float = 0.0,
        keep_alive: str | int | None = None,
        strict: bool = False,
        max_frame_failure_ratio: float = 0.25,
        on_progress: ProgressCallback | None = None,
        max_image_dimension: int = 1280,
        jpeg_quality: int = 85,
        preflight: bool = True,
        cache_dir: str | None = None,
        resume: bool = False,
    ) -> None:
        self._validate(
            min_frames,
            max_frames,
            frames_per_minute,
            request_timeout,
            context_mode,
            max_workers,
            temperature,
            max_frame_failure_ratio,
            max_image_dimension,
            jpeg_quality,
        )
        self.frame_analysis_model = frame_analysis_model
        self.summary_model = summary_model
        self.host = host.rstrip("/")
        self.api_endpoint = f"{self.host}/api/chat"
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_minute = frames_per_minute
        self.frame_selector = frame_selector or DynamicFrameSelector()
        self.audio_transcriber = audio_transcriber
        self.prompts = prompts or AnalysisPrompts()
        self.custom_frame_processor = custom_frame_processor
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.session = requests.Session()
        self.request_timeout = request_timeout
        self.request_retries = max(0, request_retries)
        self.request_backoff = max(0.0, request_backoff)
        self.retry_status_codes = {408, 429, 500, 502, 503, 504}
        self.context_max_chars = max(0, context_max_chars)
        self.audio_context_max_chars = max(0, audio_context_max_chars)
        self.context_mode = context_mode
        self.max_workers = max_workers
        self.structured_output = structured_output
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.strict = strict
        self.max_frame_failure_ratio = max_frame_failure_ratio
        self.on_progress = on_progress
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        self.preflight_enabled = preflight
        self.cache_dir = cache_dir
        self.resume = resume
        self._usage = UsageMetadata()
        self._usage_lock = threading.Lock()

    @staticmethod
    def _validate(
        min_frames: int,
        max_frames: int,
        frames_per_minute: float,
        request_timeout: float,
        context_mode: str,
        max_workers: int,
        temperature: float,
        max_frame_failure_ratio: float,
        max_image_dimension: int,
        jpeg_quality: int,
    ) -> None:
        if min_frames < 1 or max_frames < min_frames or frames_per_minute <= 0:
            raise ConfigurationError("Invalid frame-budget configuration.")
        if request_timeout <= 0 or max_workers < 1:
            raise ConfigurationError("request_timeout and max_workers must be positive.")
        if context_mode not in {"sliding", "independent"}:
            raise ConfigurationError("context_mode must be 'sliding' or 'independent'.")
        if temperature < 0:
            raise ConfigurationError("temperature must not be negative.")
        if not 0 <= max_frame_failure_ratio <= 1:
            raise ConfigurationError("max_frame_failure_ratio must be between 0 and 1.")
        if max_image_dimension < 64 or not 1 <= jpeg_quality <= 100:
            raise ConfigurationError("Invalid image preprocessing configuration.")

    def _emit(
        self,
        stage: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
        **details: object,
    ) -> None:
        if not self.on_progress:
            return
        try:
            self.on_progress(ProgressEvent(stage, current, total, message, dict(details)))
        except Exception as exc:
            self.logger.warning("Progress callback failed: %s", exc)

    def _frame_to_base64(self, frame: object) -> str:
        image = Image.fromarray(frame)
        image.thumbnail(
            (self.max_image_dimension, self.max_image_dimension),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _format_transcript(self, segments: list[AudioSegment]) -> str:
        return "\n".join(
            f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s]: {segment.text}"
            for segment in segments
        )

    def _format_frame_descriptions(self, descriptions: list[FrameAnalysis]) -> str:
        return "\n".join(
            f"Time {item.timestamp:.2f}s ({item.scene_type}; {item.selection_reason}): "
            f"{item.description}"
            for item in descriptions
        )

    def _calculate_dynamic_frame_count(
        self, video_duration: float, _scene_changes: list[float]
    ) -> int:
        """Compatibility helper; scene density no longer inflates the API/GPU budget."""
        return self._calculate_uniform_frame_count(video_duration)

    def _calculate_uniform_frame_count(self, video_duration: float) -> int:
        calculated = round(max(0.0, video_duration) / 60 * self.frames_per_minute)
        return min(self.max_frames, max(self.min_frames, calculated))

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        compact = " ".join(text.split())
        if max_chars <= 0 or len(compact) <= max_chars:
            return compact
        clipped = compact[:max_chars].rsplit(" ", 1)[0]
        return f"{clipped}..." if clipped else compact[:max_chars]

    @staticmethod
    def _is_low_signal_audio(text: str) -> bool:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if len(tokens) < 8:
            return True
        return len(set(tokens)) / len(tokens) < 0.2

    def _build_context_note(self, context: str | None, audio_text: str | None) -> str | None:
        parts = []
        if context:
            parts.append(f"Visual context: {self._truncate_text(context, self.context_max_chars)}")
        if audio_text:
            parts.append(
                f"Audio context: {self._truncate_text(audio_text, self.audio_context_max_chars)}"
            )
        if not parts:
            return None
        return (
            "Use this context only for continuity. Never claim it is visible unless the frame "
            "supports it.\n" + "\n".join(parts)
        )

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = self.request_backoff * (2 ** (attempt - 1))
        if delay > 0:
            time.sleep(delay)

    def _post_with_retries(
        self,
        payload: dict[str, Any],
        request_label: str,
        endpoint: str | None = None,
    ) -> requests.Response:
        attempts = self.request_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    endpoint or self.api_endpoint,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if response.status_code in self.retry_status_codes:
                    raise requests.exceptions.HTTPError(
                        f"Retryable HTTP status {response.status_code}", response=response
                    )
                if response.status_code == 404:
                    raise ModelNotFoundError(
                        f"Ollama {request_label} failed with 404. "
                        "Check that the model is installed."
                    )
                response.raise_for_status()
                return response
            except ModelNotFoundError:
                raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= attempts:
                    raise ProviderConnectionError(
                        f"Ollama {request_label} request failed: {exc}"
                    ) from exc
                self.logger.warning(
                    "Ollama %s request failed (attempt %s/%s): %s",
                    request_label,
                    attempt,
                    attempts,
                    exc,
                )
                self._sleep_with_backoff(attempt)
            except requests.exceptions.HTTPError as exc:
                response = exc.response
                status = response.status_code if response is not None else None
                if status in self.retry_status_codes and attempt < attempts:
                    self._sleep_with_backoff(attempt)
                    continue
                raise ProviderConnectionError(
                    f"Ollama {request_label} returned HTTP {status}: {exc}"
                ) from exc
        raise ProviderConnectionError(f"Ollama {request_label} exhausted retries.")

    def _get_json(self, endpoint: str, label: str) -> dict[str, Any]:
        try:
            response = self.session.get(endpoint, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("response was not an object")
            return data
        except Exception as exc:
            raise ProviderConnectionError(f"Unable to retrieve Ollama {label}: {exc}") from exc

    def check_environment(self) -> dict[str, Any]:
        version = self._get_json(f"{self.host}/api/version", "server version")
        tags = self._get_json(f"{self.host}/api/tags", "model list")
        installed = {
            str(item.get("name") or item.get("model"))
            for item in tags.get("models", [])
            if item.get("name") or item.get("model")
        }
        missing = [
            model
            for model in {self.frame_analysis_model, self.summary_model}
            if model not in installed
        ]
        if missing:
            raise ModelNotFoundError(
                "Required Ollama model(s) are not installed: "
                + ", ".join(sorted(missing))
                + ". Pull them explicitly with `ollama pull <model>`."
            )

        model_reports: dict[str, Any] = {}
        for model in {self.frame_analysis_model, self.summary_model}:
            response = self._post_with_retries(
                {"model": model},
                f"model inspection for {model}",
                endpoint=f"{self.host}/api/show",
            )
            model_reports[model] = response.json()
        capabilities = model_reports[self.frame_analysis_model].get("capabilities") or []
        if capabilities and "vision" not in capabilities:
            raise ModelCapabilityError(
                f"Model '{self.frame_analysis_model}' is installed but does not advertise "
                "vision support. Choose an Ollama vision model."
            )
        return {
            "ok": True,
            "version": version.get("version"),
            "installed_models": sorted(installed),
            "frame_model_capabilities": capabilities,
            "audio_enabled": self.audio_transcriber is not None,
        }

    def _payload(
        self, model: str, messages: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.structured_output:
            payload["format"] = schema
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _record_telemetry(self, result: dict[str, Any]) -> None:
        with self._usage_lock:
            prompt = int(result.get("prompt_eval_count") or 0)
            completion = int(result.get("eval_count") or 0)
            self._usage.input_tokens += prompt
            self._usage.output_tokens += completion
            self._usage.total_tokens += prompt + completion
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
                self._usage.provider_details[key] = int(
                    self._usage.provider_details.get(key, 0)
                ) + int(result.get(key) or 0)

    @staticmethod
    def _normalize_custom_analysis(
        frame: Frame, value: FrameAnalysis | Mapping[str, Any]
    ) -> FrameAnalysis:
        if isinstance(value, FrameAnalysis):
            return value
        if not isinstance(value, Mapping):
            raise ConfigurationError(
                "custom_frame_processor must return FrameAnalysis or a mapping."
            )
        if not value.get("description"):
            raise ConfigurationError(
                "custom_frame_processor mapping must include a non-empty description."
            )
        return FrameAnalysis(
            timestamp=float(value.get("timestamp", frame.timestamp)),
            description=str(value["description"]),
            scene_type=str(value.get("scene_type", frame.scene_type.value)),
            selection_reason=str(value.get("selection_reason", frame.selection_reason)),
            difference_score=float(value.get("difference_score", frame.difference_score)),
            objects=_optional_string_list(value, "objects"),
            actions=_optional_string_list(value, "actions"),
            visible_text=_optional_string_list(value, "visible_text"),
            tags=_optional_string_list(value, "tags"),
            error=str(value["error"]) if value.get("error") else None,
        )

    def _analyze_frame(self, frame: Frame, prompt_override: str | None = None) -> FrameAnalysis:
        try:
            if self.custom_frame_processor:
                return self._normalize_custom_analysis(frame, self.custom_frame_processor(frame))

            messages = []
            if self.prompts.frame_analysis_system:
                messages.append({"role": "system", "content": self.prompts.frame_analysis_system})
            messages.append(
                {
                    "role": "user",
                    "content": prompt_override or self.prompts.frame_analysis,
                    "images": [self._frame_to_base64(frame.image)],
                }
            )
            response = self._post_with_retries(
                self._payload(self.frame_analysis_model, messages, FRAME_OUTPUT_SCHEMA),
                "frame analysis",
            )
            result = response.json()
            self._record_telemetry(result)
            content = str(result.get("message", {}).get("content") or "")
            if self.structured_output:
                payload = _parse_json_object(content)
                return FrameAnalysis(
                    timestamp=frame.timestamp,
                    description=_required_text(payload, "description"),
                    scene_type=frame.scene_type.value,
                    selection_reason=frame.selection_reason,
                    difference_score=frame.difference_score,
                    objects=_required_string_list(payload, "objects"),
                    actions=_required_string_list(payload, "actions"),
                    visible_text=_required_string_list(payload, "visible_text"),
                    tags=_required_string_list(payload, "tags"),
                )
            if not content:
                raise ValueError("message content is empty")
            return FrameAnalysis(
                timestamp=frame.timestamp,
                description=content,
                scene_type=frame.scene_type.value,
                selection_reason=frame.selection_reason,
                difference_score=frame.difference_score,
            )
        except (ConfigurationError, ModelNotFoundError, ModelCapabilityError):
            raise
        except Exception as exc:
            if self.strict:
                raise
            return FrameAnalysis(
                timestamp=frame.timestamp,
                description="Error analyzing frame",
                scene_type=frame.scene_type.value,
                selection_reason=frame.selection_reason,
                difference_score=frame.difference_score,
                error=str(exc),
            )

    def _summary_prompt(self, timeline: str, transcript: str, duration: float) -> str:
        detailed = self.prompts.detailed_summary.format(
            duration=duration, timeline=timeline, transcript=transcript
        )
        brief = self.prompts.brief_summary.format(
            duration=duration, timeline=timeline, transcript=transcript
        )
        return (
            "Return one JSON result containing detailed, brief, and events. Ground every event "
            "in the supplied timestamps.\n\n"
            f"Detailed requirements:\n{detailed}\n\nBrief requirements:\n{brief}"
        )

    @staticmethod
    def _summary_from_payload(
        payload: dict[str, Any], timeline: str, transcript: str
    ) -> tuple[SummaryResult, list[TimelineEvent]]:
        summary = SummaryResult(
            detailed=_required_text(payload, "detailed"),
            brief=_required_text(payload, "brief"),
            timeline=timeline,
            transcript=transcript,
        )
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("events must be an array")
        events = []
        for item in raw_events:
            if not isinstance(item, dict):
                raise ValueError("each event must be an object")
            if not isinstance(item.get("start_time"), (int, float)) or not isinstance(
                item.get("end_time"), (int, float)
            ):
                raise ValueError("event start_time and end_time must be numbers")
            source_timestamps = item.get("source_frame_timestamps")
            if not isinstance(source_timestamps, list) or any(
                not isinstance(value, (int, float)) for value in source_timestamps
            ):
                raise ValueError("source_frame_timestamps must be an array of numbers")
            events.append(
                TimelineEvent(
                    start_time=max(0.0, float(item["start_time"])),
                    end_time=max(0.0, float(item["end_time"])),
                    description=_required_text(item, "description"),
                    source_frame_timestamps=[max(0.0, float(value)) for value in source_timestamps],
                    objects=_required_string_list(item, "objects"),
                    actions=_required_string_list(item, "actions"),
                    visible_text=_required_string_list(item, "visible_text"),
                )
            )
        return summary, events

    @staticmethod
    def _fallback_summary(
        frame_descriptions: list[FrameAnalysis], timeline: str, transcript: str
    ) -> tuple[SummaryResult, list[TimelineEvent]]:
        successful = [item for item in frame_descriptions if not item.error]
        descriptions = [item.description for item in successful]
        detailed = " ".join(descriptions) or "No frame analyses were completed successfully."
        brief = " ".join(descriptions[:2]) or detailed
        events = [
            TimelineEvent(
                item.timestamp,
                item.timestamp,
                item.description,
                [item.timestamp],
                list(item.objects),
                list(item.actions),
                list(item.visible_text),
            )
            for item in successful
        ]
        return SummaryResult(detailed, brief, timeline, transcript), events

    def _generate_summary(
        self,
        frame_descriptions: list[FrameAnalysis],
        audio_segments: list[AudioSegment],
        video_duration: float,
    ) -> tuple[SummaryResult, list[TimelineEvent], str | None]:
        timeline = self._format_frame_descriptions(frame_descriptions)
        transcript = (
            self._format_transcript(audio_segments)
            if audio_segments
            else "No audio transcript available."
        )
        payload = self._payload(
            self.summary_model,
            [
                {
                    "role": "user",
                    "content": self._summary_prompt(timeline, transcript, video_duration),
                }
            ],
            SUMMARY_OUTPUT_SCHEMA,
        )
        try:
            response = self._post_with_retries(payload, "summary")
            result = response.json()
            self._record_telemetry(result)
            content = str(result.get("message", {}).get("content") or "")
            if not self.structured_output:
                if not content.strip():
                    raise ValueError("summary content is empty")
                return SummaryResult(content, content, timeline, transcript), [], None
            try:
                summary, events = self._summary_from_payload(
                    _parse_json_object(content), timeline, transcript
                )
                return summary, events, None
            except (ValueError, TypeError, json.JSONDecodeError):
                repair = self._payload(
                    self.summary_model,
                    [
                        {
                            "role": "user",
                            "content": "Repair this invalid summary as schema-compliant JSON:\n"
                            + content,
                        }
                    ],
                    SUMMARY_OUTPUT_SCHEMA,
                )
                repaired_response = self._post_with_retries(repair, "summary repair")
                repaired_result = repaired_response.json()
                self._record_telemetry(repaired_result)
                repaired_content = str(repaired_result.get("message", {}).get("content") or "")
                summary, events = self._summary_from_payload(
                    _parse_json_object(repaired_content), timeline, transcript
                )
                return summary, events, None
        except (ModelNotFoundError, ModelCapabilityError):
            raise
        except Exception as exc:
            if self.strict:
                raise ResponseValidationError(f"Summary generation failed: {exc}") from exc
            summary, events = self._fallback_summary(frame_descriptions, timeline, transcript)
            return summary, events, f"Summary generation failed; deterministic fallback used: {exc}"

    def _sliding_frame_analysis(
        self,
        frames: list[Frame],
        audio_segments: list[AudioSegment],
        cache: AnalysisCache | None = None,
    ) -> list[FrameAnalysis]:
        descriptions: list[FrameAnalysis] = []
        context: str | None = None
        for zero_index, frame in enumerate(frames):
            index = zero_index + 1
            cached = (
                cache.read_json(f"frame_analyses/{zero_index:04d}.json")
                if cache and self.resume
                else None
            )
            if isinstance(cached, dict):
                try:
                    result = FrameAnalysis(**cached)
                    descriptions.append(result)
                    if not result.error:
                        context = result.description
                    self._emit(
                        "analyzing_frames", index, len(frames), f"Reused cached frame {index}"
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            relevant = [
                item
                for item in audio_segments
                if item.start_time <= frame.timestamp <= item.end_time
            ]
            audio_context = None
            if relevant and not self._is_low_signal_audio(relevant[0].text):
                audio_context = relevant[0].text
            note = self._build_context_note(context, audio_context)
            prompt = self.prompts.frame_analysis
            if note:
                prompt = f"{prompt}\n\n{note}"
            result = self._analyze_frame(frame, prompt)
            descriptions.append(result)
            if cache:
                cache.write_json(f"frame_analyses/{zero_index:04d}.json", asdict(result))
            if not result.error:
                context = result.description
            self._emit("analyzing_frames", index, len(frames), f"Analyzed frame {index}")
        return descriptions

    def _independent_frame_analysis(
        self, frames: list[Frame], cache: AnalysisCache | None = None
    ) -> list[FrameAnalysis]:
        descriptions: list[FrameAnalysis] = []
        pending: list[tuple[int, Frame]] = []
        for index, frame in enumerate(frames):
            cached = (
                cache.read_json(f"frame_analyses/{index:04d}.json")
                if cache and self.resume
                else None
            )
            if isinstance(cached, dict):
                try:
                    descriptions.append(FrameAnalysis(**cached))
                    continue
                except (TypeError, ValueError):
                    pass
            pending.append((index, frame))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_frame, frame): (index, frame)
                for index, frame in pending
            }
            completed = len(descriptions)
            for future in as_completed(futures):
                result = future.result()
                descriptions.append(result)
                index, _frame = futures[future]
                if cache:
                    cache.write_json(f"frame_analyses/{index:04d}.json", asdict(result))
                completed += 1
                self._emit("analyzing_frames", completed, len(frames), "Analyzed frame")
        return sorted(descriptions, key=lambda item: item.timestamp)

    def _prepare_cache(self, video_path: str) -> AnalysisCache | None:
        if not self.cache_dir:
            return None
        processor_identity = None
        if self.custom_frame_processor:
            processor_name = getattr(
                self.custom_frame_processor,
                "__qualname__",
                self.custom_frame_processor.__class__.__name__,
            )
            processor_identity = (
                f"{getattr(self.custom_frame_processor, '__module__', '')}.{processor_name}"
            )
        configuration = {
            "schema_version": "1.2",
            "provider": "ollama",
            "models": {
                "frame": self.frame_analysis_model,
                "summary": self.summary_model,
            },
            "prompts": asdict(self.prompts),
            "frame_selector": self.frame_selector.__class__.__name__,
            "selection": {
                "min_frames": self.min_frames,
                "max_frames": self.max_frames,
                "frames_per_minute": self.frames_per_minute,
                **self.frame_selector.selection_parameters(),
            },
            "image": {
                "max_image_dimension": self.max_image_dimension,
                "jpeg_quality": self.jpeg_quality,
            },
            "inference": {
                "structured_output": self.structured_output,
                "temperature": self.temperature,
                "context_mode": self.context_mode,
                "context_max_chars": self.context_max_chars,
                "audio_context_max_chars": self.audio_context_max_chars,
            },
            "audio": (
                self.audio_transcriber.__class__.__name__ if self.audio_transcriber else None
            ),
            "custom_frame_processor": processor_identity,
        }
        key = analysis_cache_key(video_path, configuration)
        cache = AnalysisCache(self.cache_dir, key)
        cache.write_json("manifest.json", {"key": key, "configuration": configuration})
        return cache

    def analyze_video_structured(self, video_path: str) -> AnalysisResult:
        if not os.path.isfile(video_path):
            raise VideoLoadError(f"Video file does not exist: {video_path}")
        self._usage = UsageMetadata()
        started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        warnings: list[str] = []
        cache = self._prepare_cache(video_path)

        if self.preflight_enabled:
            self._emit("probing", message="Checking Ollama and installed models")
            stage_started = time.perf_counter()
            self.check_environment()
            stage_seconds["preflight"] = time.perf_counter() - stage_started

        self._emit("probing", message="Probing video metadata")
        stage_started = time.perf_counter()
        video_metadata = probe_video(video_path)
        stage_seconds["probing"] = time.perf_counter() - stage_started

        self._emit("selecting_frames", message="Selecting key frames")
        stage_started = time.perf_counter()
        frames = self.frame_selector.select_frames(video_path, self)
        stage_seconds["selection"] = time.perf_counter() - stage_started
        if not frames:
            raise VideoLoadError("No decodable frames were selected.")
        if cache:
            cache.write_json("metadata.json", asdict(video_metadata))
            cache.write_json(
                "selected_frames.json",
                [
                    {
                        "timestamp": frame.timestamp,
                        "scene_type": frame.scene_type.value,
                        "selection_reason": frame.selection_reason,
                        "difference_score": frame.difference_score,
                    }
                    for frame in frames
                ],
            )

        audio_segments: list[AudioSegment] = []
        if self.audio_transcriber:
            self._emit("extracting_audio", message="Extracting audio")
            self._emit("transcribing", message="Transcribing audio")
            stage_started = time.perf_counter()
            try:
                cached_transcript = (
                    cache.read_json("transcript.json") if cache and self.resume else None
                )
                if isinstance(cached_transcript, list):
                    audio_segments = [AudioSegment(**item) for item in cached_transcript]
                else:
                    audio_segments = self.audio_transcriber.transcribe(video_path)
                    if cache:
                        cache.write_json(
                            "transcript.json", [asdict(item) for item in audio_segments]
                        )
            except Exception as exc:
                if self.strict:
                    raise
                warnings.append(f"Audio transcription failed: {exc}")
            stage_seconds["transcription"] = time.perf_counter() - stage_started

        self._emit("analyzing_frames", 0, len(frames), "Analyzing frames")
        stage_started = time.perf_counter()
        if self.context_mode == "independent":
            frame_descriptions = self._independent_frame_analysis(frames, cache=cache)
        else:
            frame_descriptions = self._sliding_frame_analysis(frames, audio_segments, cache=cache)
        stage_seconds["frame_analysis"] = time.perf_counter() - stage_started

        failed = sum(bool(item.error) for item in frame_descriptions)
        if failed:
            warnings.append(f"{failed} frame analyses failed.")
        if frame_descriptions and failed / len(frame_descriptions) > self.max_frame_failure_ratio:
            raise ResponseValidationError(
                f"Frame failure ratio {failed / len(frame_descriptions):.1%} exceeded "
                f"the configured {self.max_frame_failure_ratio:.1%}."
            )

        self._emit("summarizing", message="Generating structured summary")
        stage_started = time.perf_counter()
        summary, timeline, summary_warning = self._generate_summary(
            frame_descriptions, audio_segments, video_metadata.duration
        )
        stage_seconds["summary"] = time.perf_counter() - stage_started
        if summary_warning:
            warnings.append(summary_warning)

        scene_distribution = {
            scene_type.value: sum(frame.scene_type == scene_type for frame in frames)
            for scene_type in SceneType
        }
        selection = getattr(
            self.frame_selector,
            "last_selection_metadata",
            SelectionMetadata(
                strategy=self.frame_selector.__class__.__name__,
                selected_frame_count=len(frames),
            ),
        )
        result = AnalysisResult(
            summary=summary,
            timeline=timeline,
            frame_analyses=frame_descriptions,
            audio_segments=audio_segments,
            metadata=AnalysisMetadata(
                num_frames_analyzed=len(frames),
                num_audio_segments=len(audio_segments),
                video_duration=video_metadata.duration,
                scene_distribution=scene_distribution,
                models_used=ModelsUsed(
                    frame_analysis=self.frame_analysis_model,
                    summary=self.summary_model,
                    audio=(
                        self.audio_transcriber.__class__.__name__
                        if self.audio_transcriber
                        else None
                    ),
                    provider="ollama",
                ),
                video=video_metadata,
                selection=selection,
                performance=PerformanceMetadata(
                    total_seconds=time.perf_counter() - started,
                    stage_seconds=stage_seconds,
                ),
                usage=self._usage,
                successful_frame_analyses=len(frame_descriptions) - failed,
                failed_frame_analyses=failed,
            ),
            warnings=warnings,
            errors=[item.error for item in frame_descriptions if item.error],
        )
        self._emit("complete", len(frames), len(frames), "Video analysis complete")
        if cache:
            cache.write_json("result.json", result.to_dict())
        return result

    def analyze_video(self, video_path: str) -> dict[str, Any]:
        return self.analyze_video_structured(video_path).to_legacy_dict()
