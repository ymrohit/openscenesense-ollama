# Troubleshooting

Start with:

```bash
openscenesense-ollama --check \
  --frame-model ministral-3:latest \
  --summary-model ministral-3:latest
```

The check validates local FFmpeg tools, Ollama connectivity, installed model names, advertised
vision capability, and whether the optional audio stack is installed. It never pulls a model.

## Common failures

- **Cannot reach Ollama:** verify `OLLAMA_HOST`, network boundaries, and `/api/version`.
- **Model missing:** run `ollama list`, then explicitly `ollama pull <model>` if a download is
  intended.
- **Vision capability rejected:** choose a model whose `/api/show` response advertises `vision`.
- **Structured output rejected:** update Ollama/model first. `--no-structured-output` is a
  compatibility escape hatch and still validates parsed results.
- **Audio dependency error:** install `openscenesense-ollama[audio]`.
- **Slow or unstable concurrency:** use sliding context or `max_workers=1`.
- **Interrupted batch:** enable a cache directory and `--resume`; use `--force` to ignore cached
  stages.

Use `strict=True` during deployment validation and inspect warnings plus frame-level errors in
normal operation.
