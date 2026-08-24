# Frame selection

The default dynamic selector probes metadata, scans downscaled frames at a reduced rate, calculates
normalized grayscale differences, keeps local peaks, and suppresses nearby duplicate cuts.

Opening and closing frames are preserved. Scene changes receive no more than 60% of the remaining
budget; empty slots fill the largest temporal gaps. The budget is always bounded by `max_frames`.
Every selected frame records its reason and difference score.

Use:

- `DynamicFrameSelector` for bounded scene-aware sampling.
- `UniformFrameSelector` for reproducible temporal spacing.
- `AllFrameSelector` only for compatibility or small inputs; it intentionally decodes every frame.

The defaults are a 2 FPS scan, roughly 320-pixel-wide scan images, a `0.18` normalized difference
threshold, and a 0.75-second minimum gap. Benchmark threshold changes against representative
footage rather than raising the frame budget to compensate.
