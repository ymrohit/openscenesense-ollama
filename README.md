# OpenSceneSense Ollama

Private, local video understanding through Ollama—with a lightweight default install.

OpenSceneSense Ollama selects meaningful video frames, sends them to a local vision model, optionally transcribes audio with local Whisper, and returns summaries, timestamped events, frame-level findings, metadata, and Ollama performance telemetry.

> Prefer managed APIs and the smallest dependency footprint? Use [OpenSceneSense](https://github.com/ymrohit/openscenesense). The two packages share the v1.2 result contract while remaining intentionally separate distributions.

## The important installation split

Video analysis through Ollama does not require Torch or Transformers:

```bash
pip install openscenesense-ollama
```

Install the audio extra only when you want local Whisper transcription:

```bash
pip install "openscenesense-ollama[audio]"
```

Torch and Transformers are imported lazily when `WhisperTranscriber` is instantiated. Importing `openscenesense_ollama` or using Ollama vision does not load the audio stack.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe on `PATH`
- Ollama running locally or reachable over HTTP
- An installed vision-capable Ollama model
- A separate summary model if the vision model is not also used for summarization

OpenSceneSense never pulls a model automatically. Downloads remain an explicit user action:

```bash
ollama pull ministral-3
ollama list
```

Check the complete environment without analyzing a video:

```bash
openscenesense-ollama --check \
  --frame-model ministral-3:latest \
  --summary-model ministral-3:latest
```

The preflight retrieves the server version, verifies both models are installed, and checks advertised vision capability when the Ollama server provides it.

Model file sizes range from roughly 3 GB for compact vision models to 6 GB and beyond. See
[models, hardware, and privacy](Docs/models-and-hardware.md) for concrete starting points and the
remote-host data boundary.

## Python quick start

```python
from openscenesense_ollama import OllamaVideoAnalyzer

analyzer = OllamaVideoAnalyzer(
    frame_analysis_model="ministral-3:latest",
    summary_model="ministral-3:latest",
    min_frames=8,
    max_frames=32,
    frames_per_minute=4,
)

result = analyzer.analyze_video_structured("video.mp4")

print(result.summary.brief)
for event in result.timeline:
    print(event.start_time, event.description)

print(result.metadata.usage.provider_details)
```

`analyze_video_structured()` is the preferred v1.2 API. Existing applications can keep using the backward-compatible dictionary:

```python
legacy = analyzer.analyze_video("video.mp4")
print(legacy["brief_summary"])
```

## CLI quick start

```bash
openscenesense-ollama video.mp4 \
  --frame-model ministral-3:latest \
  --summary-model ministral-3:latest \
  --structured-output \
  --output result.json
```

Useful options:

```text
--check
--frame-selector dynamic|uniform|all
--min-frames / --max-frames / --frames-per-minute
--scene-change-threshold / --scene-scan-fps / --min-scene-gap
--temperature / --keep-alive
--context-mode sliding|independent
--context-max-chars / --audio-context-max-chars
--max-workers
--max-image-dimension / --jpeg-quality
--audio / --whisper-model / --device
--no-structured-output
--strict / --max-frame-failure-ratio
--cache-dir / --resume / --force
--structured-output
--schema
```

The CLI emits the legacy result by default for compatibility. Add `--structured-output` for the shared v1.2 schema.

## Local audio transcription

```python
from openscenesense_ollama import OllamaVideoAnalyzer, WhisperTranscriber

transcriber = WhisperTranscriber(
    model_name="openai/whisper-small",
    device="cuda:0",  # Use "cpu" when CUDA is unavailable.
    language="en",
    segment_duration=30,
)

analyzer = OllamaVideoAnalyzer(
    audio_transcriber=transcriber,
)
```

Whisper audio is extracted directly through FFmpeg as mono 16 kHz float32 samples. librosa is not required. If the audio extra is absent, the package raises an actionable installation message rather than failing during import.

CLI equivalent:

```bash
openscenesense-ollama video.mp4 --audio --whisper-model openai/whisper-small
```

## Native structured output

Frame and summary requests send JSON Schema through Ollama's `format` field with temperature `0.0` by default. The summary stage produces the detailed summary, brief summary, and timeline events in one call.

```text
AnalysisResult
├── schema_version
├── summary
│   ├── detailed
│   └── brief
├── timeline[]
├── frame_analyses[]
├── audio_segments[]
├── metadata
│   ├── video
│   ├── selection
│   ├── models
│   ├── performance
│   └── usage
├── warnings[]
└── errors[]
```

The exact schema is checked into [Docs/analysis_result.schema.json](Docs/analysis_result.schema.json) and is byte-identical to the cloud package schema.

```bash
openscenesense-ollama --schema
python scripts/export_schema.py
```

For older models that cannot enforce schemas, use `structured_output=False` or `--no-structured-output`. This lowers validation guarantees and should be treated as a compatibility mode.

## Budgeted frame selection

```python
from openscenesense_ollama import DynamicFrameSelector, OllamaVideoAnalyzer

selector = DynamicFrameSelector(
    scene_change_threshold=0.18,
    scene_scan_fps=2.0,
    min_scene_gap=0.75,
)

analyzer = OllamaVideoAnalyzer(
    frame_selector=selector,
    min_frames=8,
    max_frames=32,
    frames_per_minute=4,
)
```

The selector scans reduced-resolution frames at a limited rate, retains strong local scene-change peaks, marks opening and closing frames, and fills the largest remaining temporal gaps. It never allows scene density to inflate `max_frames`.

Every selected frame records one reason:

- `opening`
- `closing`
- `scene_change`
- `uniform_fill`

`UniformFrameSelector` gives deterministic spacing. `AllFrameSelector` is retained for compatibility and intentionally decodes every frame; it should be used carefully on long videos.

## Sequential context or independent frames

Local consumer GPUs usually perform best with sequential generation, so the default is:

```python
analyzer = OllamaVideoAnalyzer(
    context_mode="sliding",
    context_max_chars=1000,
    audio_context_max_chars=1000,
)
```

Sliding mode passes bounded prior-frame context and runs sequentially. Independent mode removes prior descriptions and can analyze frames concurrently:

```python
analyzer = OllamaVideoAnalyzer(
    context_mode="independent",
    max_workers=3,
)
```

Benchmark concurrency on your own Ollama host. More workers do not guarantee more GPU throughput.

## Custom frame processors

A custom processor may return a `FrameAnalysis` or a mapping. Mappings are normalized and must contain a non-empty `description`:

```python
from openscenesense_ollama import OllamaVideoAnalyzer


def process(frame):
    return {
        "timestamp": frame.timestamp,
        "description": "Result from another local vision pipeline",
        "objects": ["example"],
    }


analyzer = OllamaVideoAnalyzer(custom_frame_processor=process)
```

## Resilience and telemetry

```python
def progress(event):
    print(event.stage, event.current, event.total, event.message)


analyzer = OllamaVideoAnalyzer(
    request_timeout=120,
    request_retries=3,
    request_backoff=1,
    temperature=0,
    keep_alive="10m",
    strict=False,
    max_frame_failure_ratio=0.25,
    on_progress=progress,
)
```

Transient HTTP failures use bounded exponential backoff. Missing models and invalid capabilities stop immediately. Isolated stage failures become explicit warnings in normal mode; strict mode raises.

Metadata records Ollama's available `total_duration`, `load_duration`, prompt/evaluation counts, and evaluation durations. The library never calculates monetary cost.

## Cache and resume

Caching is opt-in because local results may contain sensitive descriptions and transcripts:

```python
analyzer = OllamaVideoAnalyzer(
    cache_dir=".openscenesense-cache",
    resume=True,
)
```

The cache key includes the video edge hash, size, modification nanoseconds, models, prompts, selection configuration, image preprocessing, context settings, structured-output settings, and transcription strategy. Stage files are written atomically, and completed frame analyses can be resumed after interruption.

## Development

```bash
git clone https://github.com/ymrohit/openscenesense-ollama.git
cd openscenesense-ollama
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check openscenesense_ollama tests scripts benchmarks Examples
python -m build
```

Test the optional audio environment separately:

```bash
pip install -e ".[audio,dev]"
```

CI verifies the lightweight import without Torch/Transformers, minimum and latest dependency sets, the audio extra, schema generation, supported Python versions, and wheel installation.

More detail:

- [Result schema](Docs/result-schema.md)
- [Frame selection](Docs/frame-selection.md)
- [Transcription](Docs/transcription.md)
- [Performance](Docs/performance.md)
- [Troubleshooting](Docs/troubleshooting.md)
- [Models, hardware, and privacy](Docs/models-and-hardware.md)
- [v1.2 migration](Docs/v1.2-migration.md)

## License

OpenSceneSense Ollama is released under the MIT License. Issues and contributions are welcome at [GitHub](https://github.com/ymrohit/openscenesense-ollama).
