import numpy as np

from openscenesense_ollama.transcriber import WhisperTranscriber


def test_short_audio_is_returned_as_one_segment():
    audio = np.arange(200, dtype=np.float32)

    segments = WhisperTranscriber._segment_audio(audio, 10, 30, 5)

    assert len(segments) == 1
    assert segments[0][1] == 0.0
    assert np.array_equal(segments[0][0], audio)


def test_short_tail_is_rebalanced_to_minimum_duration():
    audio = np.arange(3069, dtype=np.float32)

    segments = WhisperTranscriber._segment_audio(audio, 100, 30, 5)

    assert [len(segment) for segment, _start in segments] == [2569, 500]
    assert [start for _segment, start in segments] == [0.0, 25.69]
    assert np.array_equal(np.concatenate([segment for segment, _start in segments]), audio)


def test_exact_chunks_are_not_rebalanced():
    audio = np.arange(6000, dtype=np.float32)

    segments = WhisperTranscriber._segment_audio(audio, 100, 30, 5)

    assert [len(segment) for segment, _start in segments] == [3000, 3000]
    assert [start for _segment, start in segments] == [0.0, 30.0]
