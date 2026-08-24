# Examples

The default example requires only the lightweight installation, a running Ollama server, and an
installed vision model:

```bash
python Examples/OllamaDemo.py path/to/video.mp4 \
  --frame-model ministral-3:latest \
  --summary-model ministral-3:latest
```

Local Whisper is opt-in:

```bash
pip install -e ".[audio]"
python Examples/OllamaDemo.py path/to/video.mp4 --audio
```

`PlaygroundDemo.py` delegates to the supported package CLI, so every v1.2 option remains available:

```bash
python Examples/PlaygroundDemo.py path/to/video.mp4 --structured-output
```

Neither script downloads Ollama models automatically.
