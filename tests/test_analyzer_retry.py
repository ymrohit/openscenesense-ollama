import requests

from openscenesense_ollama.analyzer import OllamaVideoAnalyzer


class DummyResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {"message": {"content": "ok"}}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError("bad response", response=self)

    def json(self):
        return self._json_data


def test_post_with_retries_recovers_from_timeout(monkeypatch):
    analyzer = OllamaVideoAnalyzer(request_retries=2, request_backoff=0.0)
    calls = {"count": 0}

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.exceptions.ReadTimeout("timeout")
        return DummyResponse()

    monkeypatch.setattr(analyzer.session, "post", fake_post)

    response = analyzer._post_with_retries({"test": True}, "frame")
    assert response.json()["message"]["content"] == "ok"
    assert calls["count"] == 3


def test_post_with_retries_on_retryable_status(monkeypatch):
    analyzer = OllamaVideoAnalyzer(request_retries=1, request_backoff=0.0)
    calls = {"count": 0}

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResponse(status_code=503)
        return DummyResponse()

    monkeypatch.setattr(analyzer.session, "post", fake_post)

    response = analyzer._post_with_retries({"test": True}, "summary")
    assert response.json()["message"]["content"] == "ok"
    assert calls["count"] == 2
