# Models, hardware, and privacy

OpenSceneSense does not download models or reserve hardware. The frame model must accept images;
the summary model may be the same model or a smaller text-capable model.

Practical upstream-documented starting points as of the v1.2 dependency freeze:

| Model | Ollama file size | Input | Upstream minimum |
| --- | ---: | --- | --- |
| [`gemma3:4b`](https://ollama.com/library/gemma3:4b) | 3.3 GB | text, image | Ollama 0.6 |
| [`qwen3-vl:4b`](https://ollama.com/library/qwen3-vl:4b) | 3.3 GB | text, image | Ollama 0.12.7 |
| [`ministral-3:3b`](https://ollama.com/library/ministral-3:3b) | 3.0 GB | text, image | Ollama 0.13.1 |
| [`ministral-3:8b`](https://ollama.com/library/ministral-3:8b) | 6.0 GB | text, image | Ollama 0.13.1 |
| [`qwen3-vl:8b`](https://ollama.com/library/qwen3-vl:8b) | 6.1 GB | text, image | Ollama 0.12.7 |

The final local release validation used `qwen3.8:latest` exclusively for both vision and summary.
That installed Q4_K_M build reported 27.3B parameters, a 17 GB model size, a 262,144-token context,
and `completion`, `vision`, `tools`, and `thinking` capabilities. See
[live release validation](live-validation.md) for the measured video runs.

These are model-file sizes, not RAM/VRAM guarantees. Runtime memory also depends on context,
quantization, image processing, Ollama, and concurrent requests. Leave headroom above the file size
and use `ollama ps` to measure the actual target machine. Start with `max_workers=1`.

The automated suite contract-tests Ollama requests and schemas without downloading a model. Always
run `openscenesense-ollama --check` and a representative video against the exact tag you intend to
deploy.

With the default `http://localhost:11434` host, frames and prompts stay on the same machine after the
initial package/model downloads. A remote `OLLAMA_HOST` sends selected JPEGs, descriptions, and
transcript context to that remote host. Opt-in caches remain on the OpenSceneSense machine and can
contain sensitive derived content.
