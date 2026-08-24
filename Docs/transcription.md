# Optional local transcription

The default install does not include a local ML stack:

```bash
pip install openscenesense-ollama
```

Install transcription explicitly:

```bash
pip install "openscenesense-ollama[audio]"
```

Torch and Transformers are imported only when `WhisperTranscriber` is instantiated. Without the
extra, construction raises a targeted installation message; importing the package and running
Ollama vision remain lightweight.

FFmpeg extracts mono 16 kHz float32 audio directly, so librosa is not required. Pass a custom
`AudioTranscriber` to integrate another local engine, or leave `audio_transcriber=None` for visual
analysis only. Initial model downloads require network access and are never started automatically
by OpenSceneSense.

`WhisperTranscriber` selects `cuda:0` automatically when the installed Torch build exposes CUDA;
otherwise it uses CPU. `segment_duration=30` and `min_segment_duration=5` rebalance a very short
end-of-file tail into the preceding split so Whisper does not decode a sub-second chunk alone.
