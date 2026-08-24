# Performance

On the bundled 14.04-second, 351-frame development fixture, v1.2 inspected 31 scene-scan frames
(91.17% fewer than the legacy every-frame scan). A final local run measured approximately 2.24
seconds for v1.2 versus 2.65 seconds for the old selector, a 1.18× speedup. Codec, disk, CPU, and OpenCV
builds materially affect these numbers.

Reproduce the selector benchmark:

```bash
python benchmarks/benchmark_frame_selector.py Examples/pizza.mp4
```

Ollama generation usually dominates end-to-end time. Sliding context is sequential by design.
Independent context permits `max_workers > 1`, but many consumer GPU hosts serialize inference and
gain nothing from extra clients. Measure both modes on the target host.

Reduce `max_frames`, image dimensions, and context limits before sacrificing structured output.
Result metadata reports selection, transcription, frame-analysis, and summary timing plus Ollama
load/evaluation counters.
