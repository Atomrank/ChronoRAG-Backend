import pytest

from app import llm
from app.config import settings


@pytest.fixture(autouse=True)
def _azure_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "azure")


def test_no_silent_truncation(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_input_chars", 1000)
    assert len(llm._sanitize_for_azure("a" * 900)) == 900          # v1 would have cut at 14k
    with pytest.raises(llm.InputTooLongError):
        llm._sanitize_for_azure("a" * 1001)


def test_content_filter_raises_and_is_logged(monkeypatch):
    calls = []

    class Blocked(llm.BadRequestError):
        def __init__(self):
            Exception.__init__(self, "content_filter triggered")

    class FakeCompletions:
        def create(self, **kw):
            calls.append(len(kw["messages"][1]["content"]))
            raise Blocked()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "client", lambda: FakeClient())
    monkeypatch.setattr(llm, "backoff", lambda *a, **k: None)
    llm.drain_events()
    with pytest.raises(llm.ContentFilterError):
        llm.chat("sys", "x" * 5000)
    assert len(calls) == 2 and calls[0] == calls[1]                   # retried, never shrunk
    ev = llm.drain_events()
    assert [e["kind"] for e in ev] == ["content_filter", "content_filter"]


def test_finish_reason_length_is_hard_error(monkeypatch):
    """Partial completions at the output cap must not be returned."""
    class Choice:
        finish_reason = "length"
        message = type("M", (), {"content": '{"mentions":[{"partial":true}'})()

    class Resp:
        choices = [Choice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 1600})()

    class FakeCompletions:
        def create(self, **kw):
            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "client", lambda: FakeClient())
    llm.drain_events()
    with pytest.raises(llm.OutputTruncatedError):
        llm.chat("sys", "hello", max_tokens=1600,
                 call_meta={"pipeline": "kaalkram_v1", "phase": "pass1",
                            "window_index": 0})
    ev = llm.drain_events()
    kinds = [e["kind"] for e in ev]
    assert "extract_llm_call" in kinds
    assert "extract_output_truncated" in kinds
    call = next(e for e in ev if e["kind"] == "extract_llm_call")
    assert call["finish_reason"] == "length"
    assert call["max_tokens"] == 1600
    assert call["completion_tokens"] == 1600
    assert call["window_index"] == 0
