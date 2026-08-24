"""Regenerate the checked-in v1.2 result schema."""

import json
from pathlib import Path

from openscenesense_ollama.models import analysis_result_schema

destination = Path(__file__).resolve().parents[1] / "Docs" / "analysis_result.schema.json"
destination.write_text(
    json.dumps(analysis_result_schema(), indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)
