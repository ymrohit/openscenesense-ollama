# Changelog

## 1.2.0 - Unreleased

### Added

- Shared schema-versioned `AnalysisResult` contract.
- Ollama server/model/capability preflight and `--check` diagnostics.
- Native JSON Schema frame and one-call summary/event generation.
- Reduced-rate, budget-bounded dynamic frame selection.
- Sliding and independent context modes, bounded context, optional concurrency, temperature, and keep-alive controls.
- JPEG preprocessing, Ollama timing/token telemetry, progress callbacks, strict mode, and resumable stage caches.
- Mapping compatibility for custom frame processors.
- Minimum/latest dependency CI and a dedicated audio-extra job.

### Changed

- Torch and Transformers moved to the optional `[audio]` extra and are imported lazily.
- Sequential sliding-context analysis remains the default; independent frames may run concurrently.
- `analyze_video()` retains the legacy dictionary with additive telemetry.

### Removed

- Mandatory Torch, Transformers, and librosa dependencies from the default install.
- Library-level `logging.basicConfig()`.
