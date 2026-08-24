import json

from openscenesense_ollama.cli import main


def test_schema_flag(capsys):
    assert main(["--schema"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["properties"]["schema_version"]["const"] == "1.2"
