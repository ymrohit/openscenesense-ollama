"""Minimal structured Ollama example with optional local Whisper audio."""

from __future__ import annotations

import argparse

from openscenesense_ollama import OllamaVideoAnalyzer, WhisperTranscriber


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--frame-model", default="ministral-3:latest")
    parser.add_argument("--summary-model", default="ministral-3:latest")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    args = parser.parse_args()

    transcriber = WhisperTranscriber(model_name=args.whisper_model) if args.audio else None
    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=args.frame_model,
        summary_model=args.summary_model,
        min_frames=4,
        max_frames=12,
        frames_per_minute=4,
        audio_transcriber=transcriber,
    )
    result = analyzer.analyze_video_structured(args.video_path)
    print(result.summary.brief)
    for event in result.timeline:
        print(f"{event.start_time:7.2f}s  {event.description}")


if __name__ == "__main__":
    main()
