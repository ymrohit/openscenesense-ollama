"""Fail when a release artifact violates the v1.2 packaging budget."""

from __future__ import annotations

import email
import hashlib
import os
import re
import tarfile
import zipfile
from pathlib import Path

import tomllib

EXPECTED_NAME = "openscenesense-ollama"
EXPECTED_SCHEMA_SHA256 = "4ed1b7d6cc77d041e8e95ed832dd2ee0765e1ffc7fba6e262096115ea881d8fe"
MAX_WHEEL_BYTES = 250 * 1024
MAX_SDIST_BYTES = 2 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".avi", ".env", ".mp4", ".wav"}
SECRET_MARKERS = (
    b"sk-" + b"proj-",
    b"sk-" + b"or-v1-",
    b"-----BEGIN " + b"PRIVATE KEY-----",
    str(Path.home()).encode(),
)


def _forbidden(names: list[str]) -> list[str]:
    return [name for name in names if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES]


def main() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    expected_version = str(project["version"])
    init_text = Path("openscenesense_ollama/__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__ = "([^"]+)"$', init_text, re.MULTILINE)
    if not init_match or init_match.group(1) != expected_version:
        raise SystemExit("pyproject.toml and openscenesense_ollama.__version__ disagree.")
    schema_digest = hashlib.sha256(
        Path("Docs/analysis_result.schema.json").read_bytes()
    ).hexdigest()
    if schema_digest != EXPECTED_SCHEMA_SHA256:
        raise SystemExit("The v1.2 result schema does not match the shared release contract.")
    release_tag = (
        os.environ.get("GITHUB_REF_NAME") if os.environ.get("GITHUB_REF_TYPE") == "tag" else None
    )
    if release_tag and release_tag != f"v{expected_version}":
        raise SystemExit(f"Release tag {release_tag!r} does not match v{expected_version}.")

    wheels = list(Path("dist").glob("*.whl"))
    sdists = list(Path("dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution.")

    wheel, sdist = wheels[0], sdists[0]
    if wheel.stat().st_size > MAX_WHEEL_BYTES:
        raise SystemExit(f"Wheel exceeds 250 KiB: {wheel.stat().st_size} bytes")
    if sdist.stat().st_size > MAX_SDIST_BYTES:
        raise SystemExit(f"sdist exceeds 2 MiB: {sdist.stat().st_size} bytes")

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        metadata_name = next(name for name in wheel_names if name.endswith("/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
        for member in archive.infolist():
            if member.file_size <= MAX_SDIST_BYTES and any(
                marker in archive.read(member) for marker in SECRET_MARKERS
            ):
                raise SystemExit(f"Potential secret or local path found in {member.filename}.")
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()
        for member in archive.getmembers():
            if not member.isfile() or member.size > MAX_SDIST_BYTES:
                continue
            extracted = archive.extractfile(member)
            content = extracted.read() if extracted else b""
            if any(marker in content for marker in SECRET_MARKERS):
                raise SystemExit(f"Potential secret or local path found in {member.name}.")

    forbidden = _forbidden(wheel_names + sdist_names)
    if forbidden:
        raise SystemExit(f"Forbidden release files: {forbidden}")
    if metadata["Name"] != EXPECTED_NAME or metadata["Version"] != expected_version:
        raise SystemExit("Wheel metadata name/version does not match the release.")
    required = {"LICENSE", "README.md", "CHANGELOG.md"}
    packaged = {Path(name).name for name in sdist_names}
    if not required <= packaged:
        raise SystemExit(f"sdist is missing: {sorted(required - packaged)}")
    print(
        f"release artifacts OK: wheel={wheel.stat().st_size} bytes, "
        f"sdist={sdist.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
