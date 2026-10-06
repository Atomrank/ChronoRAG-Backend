"""Gemini provider: routing, key rotation, content filter and embeddings (no network)."""
import httpx
import pytest
from openai import APIStatusError, InternalServerError, RateLimitError
from pydantic import BaseModel

from app import llm
from app.config import settings


def _err(cls, status: int, message: str):
    req = httpx.Request("POST", "https://gemini.test/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=req), body=None)


class _Resp:
    def __init__(self, content="ok", finish="stop"):
        choice = type("Ch", (), {"finish_reason": finish,
                                 "message": type("M", (), {"content": content})()})()
        self.choices = [choice]
        self.usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 1})()


class _FakeGemini:
    """Per-key fake client; `script[key]` is a list of exceptions/responses consumed in order."""

    def __init__(self, key, script, calls):
        self.key = key
        outer = self

        class Completions:
            def create(self, **kw):
                calls.append((outer.key, kw))
                step = script[outer.key].pop(0) if script[outer.key] else _Resp()
                if isinstance(step, BaseException):
                    raise step
                return step

        class Embeddings:
            def create(self, **kw):
                calls.append((outer.key, kw))
                vec = [3.0, 4.0] + [0.0] * (kw["dimensions"] - 2)
                data = [type("D", (), {"embedding": vec})() for _ in kw["input"]]
                return type("R", (), {"data": data})()

        self.chat = type("C", (), {"completions": Completions()})()
        self.embeddings = Embeddings()


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_chat_model", "gem-test")
    monkeypatch.setattr(settings, "gemini_api_keys", "key-aaaa1111, key-bbbb2222\nkey-cccc3333,key-aaaa1111")
    monkeypatch.setattr(settings, "gemini_reasoning_effort", "low")
    monkeypatch.setattr(settings, "gemini_thinking_headroom", 100)
    monkeypatch.setattr(settings, "gemini_fallback_models", "")
    monkeypatch.setattr(settings, "gemini_transient_retries", 2)
    monkeypatch.setattr(settings, "gemini_max_cooldown_wait_s", 120.0)
    monkeypatch.setattr(llm, "_gemini_pool", None)
    monkeypatch.setattr(llm, "backoff", lambda *a, **k: None)
    script = {"key-aaaa1111": [], "key-bbbb2222": [], "key-cccc3333": []}
    calls: list = []
    clients = {k: _FakeGemini(k, script, calls) for k in script}
    monkeypatch.setattr(llm, "gemini_client", lambda key: clients[key])
    llm.drain_events()
    return script, calls


def test_route_prefixes(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_chat_model", "gem-default")
    monkeypatch.setattr(settings, "local_llm_model", "loc-default")
    assert llm._route(None) == ("gemini", "gem-default")
    assert llm._route("gem-other") == ("gemini", "gem-other")
    assert llm._route("azure:dep") == ("azure", "dep")
    assert llm._route("local:") == ("local", "loc-default")
    assert llm.model_id("gemini:gem-default") == llm.model_id(None)
    monkeypatch.setattr(settings, "llm_provider", "nope")
    with pytest.raises(ValueError):
        llm._route(None)


def test_keys_deduplicated_and_round_robin(gemini):
    _, calls = gemini
    for _ in range(4):
        assert llm.chat("sys", "hi") == "ok"
    assert len(llm.gemini_pool().keys) == 3
    assert [k for k, _ in calls] == ["key-aaaa1111", "key-bbbb2222", "key-cccc3333", "key-aaaa1111"]


def test_reasoning_effort_and_thinking_headroom(gemini):
    _, calls = gemini
    llm.chat("sys", "hi", max_tokens=50)
    kw = calls[0][1]
    assert kw["model"] == "gem-test"
    assert kw["reasoning_effort"] == "low"
    assert kw["max_tokens"] == 150


def test_rate_limited_key_cools_down_and_call_moves_on(gemini):
    script, calls = gemini
    script["key-aaaa1111"].append(_err(RateLimitError, 429, "quota exceeded. Please retry in 37.5s."))
    assert llm.chat("sys", "hi", retries=1) == "ok"
    assert [k for k, _ in calls] == ["key-aaaa1111", "key-bbbb2222"]
    ev = [e for e in llm.drain_events() if e["kind"] == "llm_key_rate_limited"]
    assert ev and ev[0]["cooldown_s"] == 37.5 and ev[0]["key"] == "...1111"
    llm.chat("sys", "hi")
    llm.chat("sys", "hi")
    assert "key-aaaa1111" not in [k for k, _ in calls[2:]]          # still cooling


def test_dead_keys_dropped_for_good(gemini):
    script, calls = gemini
    script["key-aaaa1111"].append(_err(APIStatusError, 402, "prepayment credits are depleted"))
    script["key-bbbb2222"].append(_err(APIStatusError, 403, "PERMISSION_DENIED"))
    for _ in range(3):
        assert llm.chat("sys", "hi", retries=1) == "ok"
    used = [k for k, _ in calls]
    assert used == ["key-aaaa1111", "key-bbbb2222", "key-cccc3333", "key-cccc3333", "key-cccc3333"]
    kinds = [e["kind"] for e in llm.drain_events()]
    assert kinds.count("llm_key_disabled") == 2
    assert llm.gemini_pool().status()["disabled"] == 2


def test_all_keys_dead_raises(gemini):
    script, _ = gemini
    for k in script:
        script[k].append(_err(APIStatusError, 401, "API key not valid"))
    with pytest.raises(llm.AllKeysUnavailableError):
        llm.chat("sys", "hi", retries=1)


def test_no_keys_configured_raises(gemini, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_keys", "")
    with pytest.raises(llm.AllKeysUnavailableError):
        llm.chat("sys", "hi", retries=1)


def test_transient_errors_retry_on_next_key(gemini, monkeypatch):
    script, calls = gemini
    monkeypatch.setattr(settings, "gemini_transient_retries", 2)
    script["key-aaaa1111"].append(_err(InternalServerError, 503, "high demand"))
    script["key-bbbb2222"].append(_err(InternalServerError, 503, "high demand"))
    assert llm.chat("sys", "hi", retries=1) == "ok"
    assert [k for k, _ in calls] == ["key-aaaa1111", "key-bbbb2222", "key-cccc3333"]
    for k in script:
        script[k].append(_err(InternalServerError, 503, "high demand"))
    with pytest.raises(InternalServerError):
        llm.chat("sys", "hi", retries=1)
    kinds = [e["kind"] for e in llm.drain_events()]
    assert kinds.count("llm_transient_error") == 2 + 3


def test_overloaded_default_model_falls_back(gemini, monkeypatch):
    script, calls = gemini
    monkeypatch.setattr(settings, "gemini_transient_retries", 2)
    monkeypatch.setattr(settings, "gemini_fallback_models", "gem-test, gem-backup")
    for k in script:
        script[k].append(_err(InternalServerError, 503, "high demand"))
    assert llm.chat("sys", "hi", retries=1) == "ok"
    assert [kw["model"] for _, kw in calls] == ["gem-test"] * 3 + ["gem-backup"]
    ev = [e for e in llm.drain_events() if e["kind"] == "llm_model_fallback"]
    assert len(ev) == 1 and ev[0]["next_model"] == "gem-backup"


def test_named_model_never_falls_back(gemini, monkeypatch):
    class Out(BaseModel):
        label: str

    script, calls = gemini
    monkeypatch.setattr(settings, "gemini_transient_retries", 2)
    monkeypatch.setattr(settings, "gemini_fallback_models", "gem-backup")
    for k in script:
        script[k].append(_err(InternalServerError, 503, "high demand"))
    with pytest.raises(InternalServerError):
        llm.chat_structured("sys", "hi", Out, deployment="gemini:gem-judge", retries=1)
    assert {kw["model"] for _, kw in calls} == {"gem-judge"}


def test_all_keys_cooling_too_long_raises(gemini, monkeypatch):
    script, _ = gemini
    monkeypatch.setattr(settings, "gemini_max_cooldown_wait_s", 1.0)
    for k in script:
        script[k].append(_err(RateLimitError, 429, "quota. Please retry in 37s."))
    with pytest.raises(llm.AllKeysUnavailableError, match="rate-limited"):
        llm.chat("sys", "hi", retries=1)


def test_rate_limit_is_per_model_and_triggers_fallback(gemini, monkeypatch):
    script, calls = gemini
    monkeypatch.setattr(settings, "gemini_max_cooldown_wait_s", 1.0)
    monkeypatch.setattr(settings, "gemini_fallback_models", "gem-backup")
    for k in script:
        script[k].append(_err(RateLimitError, 429, "daily quota. Please retry in 25000s."))
    assert llm.chat("sys", "hi", retries=1) == "ok"
    assert [kw["model"] for _, kw in calls] == ["gem-test"] * 3 + ["gem-backup"]
    assert llm.gemini_pool().status("gem-test")["cooling"] == 3
    assert llm.gemini_pool().status("gem-backup")["cooling"] == 0


def test_finish_reason_content_filter_is_logged_and_raised(gemini):
    script, calls = gemini
    script["key-aaaa1111"].append(_Resp(content=None, finish="content_filter"))
    script["key-bbbb2222"].append(_Resp(content=None, finish="content_filter"))
    with pytest.raises(llm.ContentFilterError, match="gemini"):
        llm.chat("sys", "x" * 300)
    assert len(calls) == 2
    ev = [e for e in llm.drain_events() if e["kind"] == "content_filter"]
    assert len(ev) == 2 and ev[0]["provider"] == "gemini"


def test_structured_uses_routed_model(gemini):
    class Out(BaseModel):
        label: str

    script, calls = gemini
    script["key-aaaa1111"].append(_Resp(content='{"label": "before"}'))
    out = llm.chat_structured("sys", "hi", Out, deployment="gemini:gem-judge")
    assert out.label == "before"
    kw = calls[0][1]
    assert kw["model"] == "gem-judge"
    assert kw["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_embed_requests_dimension_and_normalises(gemini, monkeypatch):
    _, calls = gemini
    monkeypatch.setattr(settings, "embed_dim", 8)
    monkeypatch.setattr(settings, "gemini_embed_model", "gem-embed")
    vecs = llm.embed(["a", "b", "c"])
    assert len(vecs) == 3 and all(len(v) == 8 for v in vecs)
    assert vecs[0][:2] == pytest.approx([0.6, 0.8])
    assert calls[0][1]["model"] == "gem-embed" and calls[0][1]["dimensions"] == 8
