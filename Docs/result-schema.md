# v1.2 result schema

`analyze_video_structured()` returns an `AnalysisResult` defined by
[`analysis_result.schema.json`](analysis_result.schema.json). The file is byte-for-byte aligned with
the cloud OpenSceneSense package.

Results contain detailed and brief summaries, grounded timeline events, frame findings, optional
audio segments, selection/model/performance/usage metadata, warnings, and errors. Ollama evaluation
and duration counters are carried in usage metadata; the library does not invent a monetary price.

Validate serialized output:

```python
from jsonschema import validate
from openscenesense_ollama import analysis_result_schema

validate(result.to_dict(), analysis_result_schema())
```

`analyze_video()` retains the v1.1 dictionary shape. Regenerate the checked-in schema with
`python scripts/export_schema.py`.
