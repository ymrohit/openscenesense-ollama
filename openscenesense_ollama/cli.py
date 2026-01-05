from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .analyzer import OllamaVideoAnalyzer
from .frame_selectors import DynamicFrameSelector, UniformFrameSelector, AllFrameSelector
from .models import AnalysisPrompts
from .transcriber import WhisperTranscriber


def _load_prompts(prompts_file: Optional[str]) -> Dict[str, str]:
    if not prompts_file:
        return {}

    with open(prompts_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Prompts file must contain a JSON object.")

    return {key: value for key, value in data.items() if value}


def _build_cache_key(video_path: str, params: Dict[str, Any]) -> str:
    stat = os.stat(video_path)
    payload = {
        "path": os.path.abspath(video_path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "params": params,
    }
    digest = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(digest).hexdigest()


def _write_output(data: Dict[str, Any], output_path: Optional[str]) -> None:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=True)
        return

    print(json.dumps(data, indent=2, ensure_ascii=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze videos using OpenSceneSense Ollama and output JSON results."
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--frame-model", default="minicpm-v")
    parser.add_argument("--summary-model", default="llama3.2")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frames-per-minute", type=float, default=4.0)
    parser.add_argument("--frame-selector", choices=["dynamic", "uniform", "all"], default="dynamic")
    parser.add_argument("--dynamic-threshold", type=float, default=20.0)
    parser.add_argument("--audio", action="store_true", help="Enable Whisper audio transcription")
    parser.add_argument("--whisper-model", default="openai/whisper-tiny")
    parser.add_argument("--device", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--prompts-file", help="JSON file with prompt templates")
    parser.add_argument("--frame-prompt")
    parser.add_argument("--detailed-prompt")
    parser.add_argument("--brief-prompt")
    parser.add_argument("--output", help="Write results to this JSON file")
    parser.add_argument("--cache-dir", help="Directory to cache analysis results")
    parser.add_argument("--force", action="store_true", help="Ignore cached results")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    prompts_data = _load_prompts(args.prompts_file)
    if args.frame_prompt:
        prompts_data["frame_analysis"] = args.frame_prompt
    if args.detailed_prompt:
        prompts_data["detailed_summary"] = args.detailed_prompt
    if args.brief_prompt:
        prompts_data["brief_summary"] = args.brief_prompt

    prompts = AnalysisPrompts(**prompts_data) if prompts_data else None

    selector_map = {
        "dynamic": DynamicFrameSelector(threshold=args.dynamic_threshold),
        "uniform": UniformFrameSelector(),
        "all": AllFrameSelector(),
    }

    audio_transcriber = None
    if args.audio:
        audio_transcriber = WhisperTranscriber(
            model_name=args.whisper_model,
            device=args.device,
        )

    params_for_cache = {
        "frame_model": args.frame_model,
        "summary_model": args.summary_model,
        "host": args.host,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "frames_per_minute": args.frames_per_minute,
        "frame_selector": args.frame_selector,
        "dynamic_threshold": args.dynamic_threshold,
        "audio": args.audio,
        "whisper_model": args.whisper_model,
        "device": args.device,
        "timeout": args.timeout,
        "retries": args.retries,
        "retry_backoff": args.retry_backoff,
        "prompts": prompts_data,
    }

    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _build_cache_key(args.video_path, params_for_cache)
        cache_path = cache_dir / f"{cache_key}.json"

        if cache_path.exists() and not args.force:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            _write_output(cached, args.output)
            return 0

    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=args.frame_model,
        summary_model=args.summary_model,
        host=args.host,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        frames_per_minute=args.frames_per_minute,
        frame_selector=selector_map[args.frame_selector],
        audio_transcriber=audio_transcriber,
        prompts=prompts,
        log_level=getattr(logging, args.log_level),
        request_timeout=args.timeout,
        request_retries=args.retries,
        request_backoff=args.retry_backoff,
    )

    results = analyzer.analyze_video(args.video_path)

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=True)

    _write_output(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
