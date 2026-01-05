from openscenesense_ollama.transcriber import _collapse_repeated_phrases


def test_collapse_repeated_phrases():
    text = "Hey Dad. Hey Dad. Hey Dad. Hey Dad. Hey Dad. Hey Dad."
    collapsed = _collapse_repeated_phrases(text, min_repeats=5, max_phrase_words=3)
    assert collapsed == "Hey Dad. (repeated 6x)"


def test_collapse_repeated_phrases_respects_threshold():
    text = "Hello. Hello. Hello."
    collapsed = _collapse_repeated_phrases(text, min_repeats=5, max_phrase_words=2)
    assert collapsed == text
