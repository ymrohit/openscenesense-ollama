# Frame-selector benchmark

Run:

```bash
python benchmarks/benchmark_frame_selector.py Examples/pizza.mp4
```

On the bundled 14.04-second, 351-frame fixture during final v1.2 validation, the selector inspected 31 frames instead of 351 (91.2% fewer) and completed approximately 1.18x faster than the v1.1 full-frame difference scan. Results vary by codec and storage; use the script for release regression checks on representative inputs.
