from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .analyzer import OllamaVideoAnalyzer
from .diagnostics import check_environment as check_local_environment
from .frame_selectors import AllFrameSelector, DynamicFrameSelector, UniformFrameSelector
from .models import AnalysisPrompts, analysis_result_schema
from .progress import ProgressEvent
from .transcriber import WhisperTranscriber


def _load_prompts(prompts_file: str | None) -> dict[str, str]:
    if not prompts_file:
        return {}
    with open(prompts_file, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Prompts file must contain a JSON object.")
    return {str(key): str(value) for key, value in data.items() if value}


def _atomic_json_write(path: str, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(value, temporary, indent=2, ensure_ascii=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _write_output(data: dict[str, Any], output_path: str | None) -> None:
    if output_path:
        _atomic_json_write(output_path, data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=True))


def _progress(event: ProgressEvent) -> None:
    suffix = f" ({event.current}/{event.total})" if event.total else ""
    print(f"[{event.stage}]{suffix} {event.message}".rstrip(), file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze videos locally with Ollama and optional Whisper transcription."
    )
    parser.add_argument("video_path", nargs="?", help="Path to the video file")
    parser.add_argument("--frame-model", default="ministral-3:latest")
    parser.add_argument("--summary-model", default="ministral-3:latest")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frames-per-minute", type=float, default=4.0)
    parser.add_argument(
        "--frame-selector", choices=["dynamic", "uniform", "all"], default="dynamic"
    )
    parser.add_argument(
        "--scene-change-threshold",
        "--dynamic-threshold",
        dest="scene_change_threshold",
        type=float,
        default=0.18,
    )
    parser.add_argument("--scene-scan-fps", type=float, default=2.0)
    parser.add_argument("--min-scene-gap", type=float, default=0.75)
    parser.add_argument("--audio", action="store_true", help="Enable local Whisper transcription")
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--device")
    parser.add_argument("--audio-segment-duration", type=int, default=30)
    parser.add_argument("--audio-min-segment-duration", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-alive")
    parser.add_argument("--context-mode", choices=["sliding", "independent"], default="sliding")
    parser.add_argument("--context-max-chars", type=int, default=1000)
    parser.add_argument("--audio-context-max-chars", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-image-dimension", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--no-structured-output", action="store_true")
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Emit the v1.2 structured result instead of the legacy dictionary",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-frame-failure-ratio", type=float, default=0.25)
    parser.add_argument("--prompts-file")
    parser.add_argument("--frame-prompt")
    parser.add_argument("--detailed-prompt")
    parser.add_argument("--brief-prompt")
    parser.add_argument("--output")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        if args.schema:
            print(json.dumps(analysis_result_schema(), indent=2, ensure_ascii=True))
            return 0
        if not args.video_path and not args.check:
            parser.error("video_path is required unless --schema or --check is set.")

        prompts_data = _load_prompts(args.prompts_file)
        if args.frame_prompt:
            prompts_data["frame_analysis"] = args.frame_prompt
        if args.detailed_prompt:
            prompts_data["detailed_summary"] = args.detailed_prompt
        if args.brief_prompt:
            prompts_data["brief_summary"] = args.brief_prompt
        prompts = AnalysisPrompts(**prompts_data) if prompts_data else None

        threshold = args.scene_change_threshold
        if threshold > 1:
            threshold /= 255
        selectors = {
            "dynamic": DynamicFrameSelector(
                scene_change_threshold=threshold,
                scene_scan_fps=args.scene_scan_fps,
                min_scene_gap=args.min_scene_gap,
            ),
            "uniform": UniformFrameSelector(),
            "all": AllFrameSelector(),
        }

        audio_transcriber = None
        if args.audio and not args.check:
            audio_transcriber = WhisperTranscriber(
                model_name=args.whisper_model,
                device=args.device,
                segment_duration=args.audio_segment_duration,
                min_segment_duration=args.audio_min_segment_duration,
            )

        analyzer = OllamaVideoAnalyzer(
            frame_analysis_model=args.frame_model,
            summary_model=args.summary_model,
            host=args.host,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            frames_per_minute=args.frames_per_minute,
            frame_selector=selectors[args.frame_selector],
            audio_transcriber=audio_transcriber,
            prompts=prompts,
            log_level=getattr(logging, args.log_level),
            request_timeout=args.timeout,
            request_retries=args.retries,
            request_backoff=args.retry_backoff,
            context_max_chars=args.context_max_chars,
            audio_context_max_chars=args.audio_context_max_chars,
            context_mode=args.context_mode,
            max_workers=args.max_workers,
            structured_output=not args.no_structured_output,
            temperature=args.temperature,
            keep_alive=args.keep_alive,
            strict=args.strict,
            max_frame_failure_ratio=args.max_frame_failure_ratio,
            on_progress=None if args.quiet else _progress,
            max_image_dimension=args.max_image_dimension,
            jpeg_quality=args.jpeg_quality,
            preflight=not args.no_preflight,
            cache_dir=args.cache_dir,
            resume=args.resume and not args.force,
        )

        if args.check:
            report = check_local_environment()
            report["ollama"] = analyzer.check_environment()
            report["audio_extra_installed"] = bool(
                importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
            )
            report["ok"] = bool(report["ok"] and report["ollama"]["ok"])
            print(json.dumps(report, indent=2, ensure_ascii=True))
            return 0 if report["ok"] else 1

        result = analyzer.analyze_video_structured(args.video_path)
        payload = result.to_dict() if args.structured_output else result.to_legacy_dict()
        _write_output(payload, args.output)
        return 0
    except Exception as exc:
        print(f"openscenesense-ollama: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
