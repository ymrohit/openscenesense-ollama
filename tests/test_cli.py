import json

from openscenesense_ollama.cli import _parser, main


def test_schema_flag(capsys):
    assert main(["--schema"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["properties"]["schema_version"]["const"] == "1.2"


def test_audio_chunking_arguments():
    args = _parser().parse_args(
        [
            "video.mp4",
            "--audio",
            "--audio-segment-duration",
            "20",
            "--audio-min-segment-duration",
            "4.5",
        ]
    )

    assert args.audio is True
    assert args.audio_segment_duration == 20
    assert args.audio_min_segment_duration == 4.5
