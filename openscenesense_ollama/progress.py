from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    current: int = 0
    total: int = 0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
