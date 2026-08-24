from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def video_fingerprint(video_path: str, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    path = Path(video_path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "edge_sha256": digest.hexdigest(),
    }


def analysis_cache_key(video_path: str, configuration: dict[str, Any]) -> str:
    payload = {
        "video": video_fingerprint(video_path),
        "configuration": configuration,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class AnalysisCache:
    def __init__(self, cache_dir: str | os.PathLike[str], key: str) -> None:
        self.directory = Path(cache_dir) / key

    def path(self, relative_path: str) -> Path:
        candidate = self.directory / relative_path
        if self.directory.resolve() not in candidate.resolve().parents:
            raise ValueError("Cache path escapes the analysis directory.")
        return candidate

    def read_json(self, relative_path: str) -> Any | None:
        path = self.path(relative_path)
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def write_json(self, relative_path: str, value: Any) -> None:
        destination = self.path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                json.dump(value, temporary, indent=2, ensure_ascii=True)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
