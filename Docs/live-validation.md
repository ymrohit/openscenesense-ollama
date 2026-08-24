# Live release validation

These checks were run on 2026-08-24 against a local Ollama 0.32.14 server and an NVIDIA GeForce RTX
4090. They supplement the unit, minimum/latest dependency, schema, and packaging matrices.

## Real audiovisual fixtures

- The full 596.46-second Blender Foundation *Big Buck Bunny* MP4 was downloaded from
  `https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4.zip`.
- The extracted 64,657,027-byte MP4 had SHA-256
  `f78f39603e6774907f2faafabf26a6674a6fc31769ec304a8a8f7c62d280508`, H.264 video, and stereo
  AAC audio.
- The bundled 30.69-second `Examples/genvideo.mp4` supplied narrated speech coverage.

The long fixture and downloaded model weights are intentionally excluded from release artifacts.

## Qwen 3.8 vision and summary

`qwen3.8:latest` was used exclusively for both visual frame analysis and final summarization. The
local model advertised `completion`, `vision`, `tools`, and `thinking` capabilities.

The full film run analyzed ten dynamically selected frames and generated ten timeline events in
83.08 seconds. The v1.2 result validated successfully, recorded `qwen3.8:latest` for both model
roles, and contained no warnings or errors. Ollama reported 11,437 evaluated/prompt tokens in
total.

## GPU Whisper and combined pipeline

The audio extra was installed with Torch 2.13.0+cu130 and Transformers 5.15.1. The transcriber
auto-selected `cuda:0`, loaded the default `openai/whisper-small`, and processed the narrated
fixture on the RTX 4090.

The first run exposed a hallucinated word from decoding a final 0.69-second chunk alone. v1.2 now
rebalances such tails to a configurable minimum duration. The repeated run produced two coherent
segments spanning 0.00-25.69 and 25.69-30.69 seconds.

The final combined run used Qwen 3.8 for four visual frames and summary plus GPU Whisper for audio.
It produced two audio segments and three events in 50.92 seconds, validated against the shared
schema, with no warnings or errors.
