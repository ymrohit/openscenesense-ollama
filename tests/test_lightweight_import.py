import builtins
import subprocess
import sys

import pytest

from openscenesense_ollama.exceptions import MissingDependencyError
from openscenesense_ollama.transcriber import WhisperTranscriber


def test_default_import_does_not_load_audio_stack():
    code = (
        "import sys, openscenesense_ollama; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_whisper_reports_targeted_audio_extra(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformers"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(MissingDependencyError, match="openscenesense-ollama\\[audio\\]"):
        WhisperTranscriber()
