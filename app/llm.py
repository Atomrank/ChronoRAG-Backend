import copy
import json
import random
import re
import threading
import time
from typing import Any

import numpy as np
from openai import (AzureOpenAI, OpenAI, APIStatusError, APITimeoutError, APIConnectionError,
                    BadRequestError)

from .config import settings

_client: AzureOpenAI | None = None
_lock = threading.Lock()

# Cumulative token accounting, per-thread-safe
_usage = {"prompt": 0, "completion": 0}


class ContentFilterError(RuntimeError):
    """The provider blocked the prompt/response under its content policy."""


class AllKeysUnavailableError(RuntimeError):
    """No usable API key is left: none configured, or every key is revoked/unfunded."""


class KeysRateLimitedError(AllKeysUnavailableError):
    """Every live key is rate-limited for this model for longer than we are willing to wait."""


class _FilteredResponse(Exception):
    """A completion came back with finish_reason=content_filter instead of an HTTP error."""


class InputTooLongError(ValueError):
    """Input exceeds settings.llm_max_input_chars. Callers must window the text;
    it is never cut silently."""


class OutputTruncatedError(RuntimeError):
    """Model hit the output token cap (finish_reason length/max_tokens).
    Partial completions must not be accepted as valid extraction results."""


# Build-health events (content-filter hits, degraded retries, extract call
# telemetry). Callers drain these and persist them to build_events so data loss
# is always visible.
_events: list[dict] = []


def _log_event(kind: str, **detail) -> None:
    with _lock:
        _events.append({"kind": kind, **detail})


def drain_events() -> list[dict]:
    with _lock:
        out = list(_events)
        _events.clear()
    return out


_LENGTH_FINISH = frozenset({"length", "max_tokens"})


def _finish_reason(choice) -> str | None:
    """Azure uses finish_reason; some OpenAI-compatible servers use stop_reason."""
    fr = getattr(choice, "finish_reason", None)
    if fr:
        return str(fr)
    sr = getattr(choice, "stop_reason", None)
    return str(sr) if sr else None


def _count_structured_items(obj) -> int | None:
    """Best-effort item count for extraction telemetry (schema-agnostic)."""
    for attr in ("mentions", "instructions", "events", "items", "observations",
                 "aliases", "relations", "frame_ops"):
        val = getattr(obj, attr, None)
        if isinstance(val, list):
            return len(val)
    return None


def _count_bullet_items(text: str) -> int:
    n = 0
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith(("-", "*", "•")) or s.startswith("[PAGE"):
            n += 1
    return n


def _salvage_json_text(raw: str) -> tuple[str, bool]:
    """Strip markdown fences / leading junk. Returns (text, did_salvage)."""
    text = (raw or "").strip()
    if not text:
        return text, False
    salvaged = False
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)
        salvaged = True
    if not text.lstrip().startswith(("{", "[")):
        m = re.search(r"[\{\[]", text)
        if m:
            text = text[m.start():]
            salvaged = True
    return text.strip(), salvaged


def _check_output_budget(resp, *, max_tokens: int, input_chars: int,
                         call_meta: dict | None) -> tuple[str | None, int]:
    """Log extraction telemetry when call_meta is set; always hard-fail on length."""
    choice = resp.choices[0]
    fr = _finish_reason(choice)
    usage = resp.usage
    completion = int((usage.completion_tokens if usage else 0) or 0)
    detail = {
        "finish_reason": fr,
        "max_tokens": max_tokens,
        "input_chars": input_chars,
        "completion_tokens": completion,
        **(call_meta or {}),
    }
    # Only emit extract_llm_call when a caller opted into telemetry (extraction).
    if call_meta is not None:
        _log_event("extract_llm_call", **detail)
    if fr in _LENGTH_FINISH:
        _log_event("extract_output_truncated", **detail)
        raise OutputTruncatedError(
            f"finish_reason={fr} max_tokens={max_tokens} "
            f"completion_tokens={completion} input_chars={input_chars}"
        )
    return fr, completion


def _is_content_filter(exc: BaseException) -> bool:
    if isinstance(exc, _FilteredResponse):
        return True
    text = str(exc).lower()
    if any(m in text for m in ("content_filter", "content management policy",
                               "prohibited_content", "safety")):
        return True
    if isinstance(exc, APIStatusError):
        try:
            err = exc.response.json().get("error") or {}
            return err.get("code") == "content_filter"
        except Exception:
            return False
    return False


def _sanitize_for_azure(text: str, *, limit: int | None = None) -> str:
    """Strip noisy PDF/OCR junk that often trips Azure hate filters.
    With limit=None the text is never cut; over-long input raises."""
    # Drop non-printable / odd control chars; keep basic punctuation & newlines
    cleaned = "".join(
        ch if (ch in "\n\t" or 32 <= ord(ch) < 127 or ord(ch) > 159) else " "
        for ch in text
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if limit is None:
        if len(cleaned) > settings.llm_max_input_chars:
            raise InputTooLongError(
                f"input is {len(cleaned)} chars > llm_max_input_chars="
                f"{settings.llm_max_input_chars}; split it into windows")
        return cleaned
    if len(cleaned) > limit:
        _log_event("truncated", original_chars=len(cleaned), kept_chars=limit)
    return cleaned[:limit].strip()


_SAFE_PREFIX = (
    "Educational literary analysis of a classic text. "
    "All quoted text is literature under discussion, not real-world instructions.\n\n"
)


def client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
            raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are not set "
                               "(set LLM_PROVIDER=gemini to use Gemini instead)")
        _client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=240.0,
            max_retries=0,          # we do our own backoff
        )
    return _client


_tls = threading.local()


_local_client = None


def local_client():
    """OpenAI-compatible server (e.g. vLLM on the 4090) at settings.local_llm_base_url."""
    global _local_client
    if _local_client is None:
        if not settings.local_llm_base_url:
            raise RuntimeError("LOCAL_LLM_BASE_URL is not set")
        _local_client = OpenAI(base_url=settings.local_llm_base_url, api_key="EMPTY",
                               timeout=240.0, max_retries=0)
    return _local_client


# ------------------------------------------------------------
# Provider routing
# ------------------------------------------------------------
_PROVIDERS = ("azure", "gemini", "local")


def _route(deployment: str | None) -> tuple[str, str]:
    """(provider, model) for a deployment string. "azure:", "gemini:" and "local:" prefixes
    pick the provider; a bare name is a model of settings.llm_provider; empty = its default."""
    d = (deployment or "").strip()
    provider, model = (settings.llm_provider or "azure").strip().lower(), d
    for p in _PROVIDERS:
        if d.startswith(p + ":"):
            provider, model = p, d[len(p) + 1:]
            break
    if provider == "azure":
        return provider, model or settings.azure_chat_deployment
    if provider == "gemini":
        return provider, model or settings.gemini_chat_model
    if provider == "local":
        return provider, model or settings.local_llm_model
    raise ValueError(f"unknown LLM provider {provider!r}; expected one of {_PROVIDERS}")


def model_id(deployment: str | None = None) -> str:
    """Normalised "provider:model" for reports and same-model checks."""
    provider, model = _route(deployment)
    return f"{provider}:{model}"


def active_models() -> dict:
    provider, model = _route(None)
    embed_model = (settings.gemini_embed_model if provider == "gemini"
                   else settings.azure_embed_deployment)
    return {"provider": provider, "chat_model": model, "embed_model": embed_model,
            "embed_dim": settings.embed_dim}


# ------------------------------------------------------------
# Gemini key rotation
# ------------------------------------------------------------
def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 8 else "***"


class KeyPool:
    """Thread-safe round-robin over API keys. Rate limits are per (key, scope) because
    provider quotas are per model; a revoked or unfunded key is dropped for every scope
    for the process lifetime."""

    def __init__(self, keys: list[str]):
        self.keys = list(dict.fromkeys(k.strip() for k in keys if k and k.strip()))
        self._next = 0
        self._cool_until: dict[tuple[str, str], float] = {}
        self._dead: dict[str, str] = {}
        self._lock = threading.Lock()

    def acquire(self, scope: str = "") -> str:
        while True:
            with self._lock:
                live = [k for k in self.keys if k not in self._dead]
                if not live:
                    reasons = {_mask(k): r for k, r in self._dead.items()}
                    raise AllKeysUnavailableError(
                        f"no usable Gemini API key ({len(self.keys)} configured; "
                        f"disabled: {reasons}). Set GEMINI_API_KEYS.")
                now = time.monotonic()
                n = len(self.keys)
                for i in range(n):
                    k = self.keys[(self._next + i) % n]
                    if k not in self._dead and self._cool_until.get((k, scope), 0.0) <= now:
                        self._next = (self._next + i + 1) % n
                        return k
                wait = min(self._cool_until[(k, scope)] for k in live) - now
            _log_event("llm_keys_all_cooling", provider="gemini", scope=scope,
                       wait_s=round(wait, 1))
            if wait > settings.gemini_max_cooldown_wait_s:
                raise KeysRateLimitedError(
                    f"all {len(live)} usable Gemini keys are rate-limited on {scope or 'this'} "
                    f"model for >= {wait:.0f}s "
                    f"(GEMINI_MAX_COOLDOWN_WAIT_S={settings.gemini_max_cooldown_wait_s:.0f})")
            time.sleep(max(0.05, wait))

    def cool(self, key: str, seconds: float, scope: str = "") -> None:
        with self._lock:
            self._cool_until[(key, scope)] = time.monotonic() + seconds

    def kill(self, key: str, reason: str) -> None:
        with self._lock:
            self._dead[key] = reason

    def status(self, scope: str = "") -> dict:
        now = time.monotonic()
        with self._lock:
            return {"keys": len(self.keys), "disabled": len(self._dead),
                    "cooling": sum(1 for k in self.keys if k not in self._dead
                                   and self._cool_until.get((k, scope), 0.0) > now)}


_gemini_pool: KeyPool | None = None
_gemini_pool_src: str | None = None
_gemini_clients: dict[str, OpenAI] = {}


def gemini_pool() -> KeyPool:
    global _gemini_pool, _gemini_pool_src
    raw = settings.gemini_api_keys or ""
    with _lock:
        if _gemini_pool is None or _gemini_pool_src != raw:
            _gemini_pool = KeyPool(re.split(r"[,\s]+", raw))
            _gemini_pool_src = raw
        return _gemini_pool


def gemini_client(key: str) -> OpenAI:
    with _lock:
        cli = _gemini_clients.get(key)
        if cli is None:
            cli = OpenAI(base_url=settings.gemini_base_url, api_key=key,
                         timeout=settings.gemini_timeout_s, max_retries=0)
            _gemini_clients[key] = cli
        return cli


def _key_fault(exc: BaseException) -> str | None:
    """"dead" = the key itself is unusable (revoked, denied, out of credit);
    "rate" = quota / rate limit on this key; None = not a key problem."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status in (401, 402, 403) or "api key not valid" in text or "api_key_invalid" in text:
        return "dead"
    if status == 429:
        return "rate"
    return None


def _retry_delay(exc: BaseException) -> float | None:
    m = re.search(r"retry(?:delay['\"]?\s*:\s*['\"]?|\s+in\s+)(\d+(?:\.\d+)?)s", str(exc), re.I)
    return float(m.group(1)) if m else None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    return (getattr(exc, "status_code", None) or 0) >= 500


def _gemini_call(fn, scope: str = ""):
    """Run fn(client) on the next usable key. Key faults rotate to another key without
    consuming the caller's retries; 5xx/timeouts retry on the next key with a short backoff.
    `scope` is the model name: rate-limit cooldowns are tracked per (key, model)."""
    pool = gemini_pool()
    rate_hits = transient = 0
    while True:
        key = pool.acquire(scope)
        try:
            return fn(gemini_client(key))
        except (APIStatusError, APITimeoutError, APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            fault = _key_fault(exc)
            if fault == "dead":
                pool.kill(key, f"{status}: {str(exc)[:120]}")
                _log_event("llm_key_disabled", provider="gemini", key=_mask(key),
                           status=status, error=str(exc)[:300], **pool.status(scope))
                continue
            if fault == "rate":
                rate_hits += 1
                delay = _retry_delay(exc) or settings.gemini_rate_limit_cooldown_s
                pool.cool(key, delay, scope)
                _log_event("llm_key_rate_limited", provider="gemini", key=_mask(key),
                           model=scope, cooldown_s=delay, **pool.status(scope))
                if rate_hits > 2 * len(pool.keys):
                    raise
                continue
            if _is_transient(exc):
                transient += 1
                _log_event("llm_transient_error", provider="gemini", key=_mask(key),
                           model=scope, status=status, error=str(exc)[:200],
                           attempt=transient)
                if transient > settings.gemini_transient_retries:
                    raise
                backoff(transient - 1, base=2.0, cap=30.0)
                continue
            raise


def _complete(provider: str, model: str, **kw):
    """One chat completion on the routed provider (no retry policy beyond key rotation)."""
    if provider == "gemini":
        kw["max_tokens"] = kw["max_tokens"] + max(0, settings.gemini_thinking_headroom)
        if settings.gemini_reasoning_effort:
            kw["reasoning_effort"] = settings.gemini_reasoning_effort
        models = [model]
        if model == settings.gemini_chat_model:
            models += [m.strip() for m in settings.gemini_fallback_models.split(",")
                       if m.strip() and m.strip() != model]
        for i, m in enumerate(models):
            try:
                return _gemini_call(lambda cli: cli.chat.completions.create(model=m, **kw),
                                    scope=m)
            except (APIStatusError, APITimeoutError, APIConnectionError,
                    KeysRateLimitedError) as exc:
                overloaded = (isinstance(exc, KeysRateLimitedError) or _is_transient(exc)
                              or getattr(exc, "status_code", None) == 429)
                if i == len(models) - 1 or not overloaded:
                    raise
                _log_event("llm_model_fallback", provider="gemini", model=m,
                           next_model=models[i + 1], error=str(exc)[:200])
    cli = local_client() if provider == "local" else client()
    return cli.chat.completions.create(model=model, **kw)


def _raise_if_filtered(resp) -> None:
    if _finish_reason(resp.choices[0]) == "content_filter":
        raise _FilteredResponse("finish_reason=content_filter")


def _filter_error(provider: str) -> ContentFilterError:
    hint = (" In Azure AI Foundry → your deployment → Content filter, set Hate/Violence to "
            "Annotate or lowest block level, then retry." if provider == "azure" else "")
    return ContentFilterError(f"{provider} content filter blocked this literary prompt.{hint}")


def _record(usage) -> None:
    if usage is None:
        return
    with _lock:
        _usage["prompt"] += usage.prompt_tokens or 0
        _usage["completion"] += usage.completion_tokens or 0
    # per-thread counters so concurrent questions do not mix their token counts
    _tls.prompt = getattr(_tls, "prompt", 0) + (usage.prompt_tokens or 0)
    _tls.completion = getattr(_tls, "completion", 0) + (usage.completion_tokens or 0)


def usage_snapshot() -> dict:
    """Token counts for the CURRENT thread (diff two snapshots around a call)."""
    return {"prompt": getattr(_tls, "prompt", 0), "completion": getattr(_tls, "completion", 0)}


def global_usage() -> dict:
    with _lock:
        return dict(_usage)


def reset_usage() -> None:
    with _lock:
        _usage["prompt"] = 0
        _usage["completion"] = 0


def backoff(attempt: int, base: float = 4.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0, 2)
    time.sleep(delay)


# ------------------------------------------------------------
# Azure Structured Outputs requires a stricter JSON Schema than
# Pydantic emits by default. This patcher makes it acceptable.
# ------------------------------------------------------------
def make_strict_schema(schema: dict) -> dict:
    """
    Azure strict mode requires, on EVERY object node including nested $defs:
      - "additionalProperties": false
      - every property listed in "required"
    It also rejects sibling keywords next to "$ref".
    """
    schema = copy.deepcopy(schema)

    def patch(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict):
                    node["required"] = list(props.keys())
            if "$ref" in node:
                for k in list(node.keys()):
                    if k != "$ref":
                        del node[k]
            for v in list(node.values()):
                patch(v)
        elif isinstance(node, list):
            for item in node:
                patch(item)

    patch(schema)
    return schema


# ------------------------------------------------------------
# Chat
# ------------------------------------------------------------
def chat(system: str, user: str, *, temperature: float = 0.1,
         max_tokens: int = 1500, retries: int = 3,
         degrade_on_filter: bool = False,
         call_meta: dict | None = None) -> str:
    """degrade_on_filter=False (default): a content-filter block raises
    ContentFilterError after one retry instead of silently shrinking the input.
    Only interactive query paths should pass True.

    call_meta: optional extraction telemetry (window_index, window_start/end, …).
    finish_reason length/max_tokens raises OutputTruncatedError (hard error).
    """
    payloads = [(system, _SAFE_PREFIX + _sanitize_for_azure(user))]
    if degrade_on_filter:
        payloads += [
            (system, _SAFE_PREFIX + _sanitize_for_azure(user, limit=4500)),
            (
                "You answer brief educational questions about classic literature using only the notes given.",
                _SAFE_PREFIX + _sanitize_for_azure(user, limit=2000),
            ),
        ]
    provider, model = _route(None)
    last_exc: BaseException | None = None
    for attempt in range(retries):
        sys_msg, user_msg = payloads[min(attempt, len(payloads) - 1)]
        try:
            resp = _complete(
                provider, model,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _record(resp.usage)
            _raise_if_filtered(resp)
            meta = None if call_meta is None else {**call_meta, "attempt": attempt}
            _check_output_budget(
                resp, max_tokens=max_tokens, input_chars=len(user_msg), call_meta=meta)
            text = (resp.choices[0].message.content or "").strip()
            # Amend last extract_llm_call with item count (bullets for pass-1 skim).
            if meta is not None:
                with _lock:
                    for ev in reversed(_events):
                        if ev.get("kind") == "extract_llm_call" and "n_items" not in ev:
                            ev["n_items"] = _count_bullet_items(text)
                            ev["json_salvaged"] = False
                            break
            return text
        except OutputTruncatedError:
            raise
        except (BadRequestError, _FilteredResponse) as exc:
            last_exc = exc
            if _is_content_filter(exc):
                _log_event("content_filter", attempt=attempt,
                           degraded=attempt > 0 and degrade_on_filter,
                           input_chars=len(user_msg), provider=provider)
                if attempt == retries - 1 or not degrade_on_filter and attempt >= 1:
                    raise _filter_error(provider) from exc
                backoff(min(attempt, 1))
                continue
            if attempt == retries - 1:
                raise
            backoff(attempt)
        except (APIStatusError, APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            backoff(attempt)
    if last_exc:
        raise last_exc
    return ""


def chat_structured(system: str, user: str, model_cls, *,
                    temperature: float = 0.1, max_tokens: int = 6000,
                    retries: int = 3, degrade_on_filter: bool = False,
                    deployment: str | None = None, max_input_chars: int | None = None,
                    call_meta: dict | None = None):
    """Return an instance of model_cls, guaranteed to validate.

    call_meta: optional extraction telemetry. finish_reason length/max_tokens
    raises OutputTruncatedError before any parse of a partial body.
    """
    schema = make_strict_schema(model_cls.model_json_schema())
    if max_input_chars and len(user) > max_input_chars:
        raise InputTooLongError(f"input is {len(user)} chars > {max_input_chars}")
    first = (_sanitize_for_azure(user, limit=len(user) + 1) if max_input_chars
             else _sanitize_for_azure(user))
    payloads = [(system, _SAFE_PREFIX + first)]
    if degrade_on_filter:
        payloads += [
            (system, _SAFE_PREFIX + _sanitize_for_azure(user, limit=4500)),
            (
                "Educational literature timeline assistant. Answer using only supplied notes.",
                _SAFE_PREFIX + _sanitize_for_azure(user, limit=2000),
            ),
        ]
    provider, model = _route(deployment)
    for attempt in range(retries):
        sys_msg, user_msg = payloads[min(attempt, len(payloads) - 1)]
        try:
            resp = _complete(
                provider, model,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_cls.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            _record(resp.usage)
            _raise_if_filtered(resp)
            meta = None if call_meta is None else {**call_meta, "attempt": attempt}
            _check_output_budget(
                resp, max_tokens=max_tokens, input_chars=len(user_msg), call_meta=meta)
            raw = resp.choices[0].message.content
            if not raw:
                raise ValueError("empty structured response")
            salvaged = False
            try:
                obj = model_cls.model_validate_json(raw)
            except Exception as parse_exc:
                repaired, did = _salvage_json_text(raw)
                if not did:
                    raise parse_exc
                try:
                    obj = model_cls.model_validate_json(repaired)
                    salvaged = True
                except Exception:
                    raise parse_exc
            if meta is not None:
                with _lock:
                    for ev in reversed(_events):
                        if ev.get("kind") == "extract_llm_call" and "n_items" not in ev:
                            ev["n_items"] = _count_structured_items(obj)
                            ev["json_salvaged"] = bool(salvaged)
                            break
            return obj
        except OutputTruncatedError:
            raise
        except (BadRequestError, _FilteredResponse) as exc:
            if _is_content_filter(exc):
                _log_event("content_filter", attempt=attempt,
                           degraded=attempt > 0 and degrade_on_filter,
                           input_chars=len(user_msg), provider=provider)
                if attempt == retries - 1 or not degrade_on_filter and attempt >= 1:
                    raise _filter_error(provider) from exc
                backoff(min(attempt, 1))
                continue
            if attempt == retries - 1:
                raise
            backoff(attempt)
        except (APIStatusError, APITimeoutError, APIConnectionError, ValueError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            backoff(attempt)
    raise RuntimeError("unreachable")


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------
def _unit(vec: list[float]) -> list[float]:
    """Gemini vectors at reduced dimensionality are not unit-length; Azure's are."""
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Batch-embed. Providers cap a single request; we chunk at 96 inputs."""
    provider = _route(None)[0]
    out: list[list[float]] = []
    for i in range(0, len(texts), 96):
        window = [t.replace("\n", " ")[:8000] for t in texts[i:i + 96]]
        for attempt in range(retries):
            try:
                if provider == "gemini":
                    resp = _gemini_call(lambda cli: cli.embeddings.create(
                        model=settings.gemini_embed_model, input=window,
                        dimensions=settings.embed_dim), scope=settings.gemini_embed_model)
                    vecs = [_unit(d.embedding) for d in resp.data]
                elif provider == "azure":
                    resp = client().embeddings.create(
                        model=settings.azure_embed_deployment,
                        input=window,
                    )
                    vecs = [d.embedding for d in resp.data]
                else:
                    raise ValueError(f"provider {provider!r} has no embedding endpoint")
                if len(vecs) != len(window) or any(len(v) != settings.embed_dim for v in vecs):
                    raise RuntimeError(
                        f"embedding response mismatch: {len(vecs)} vectors for {len(window)} "
                        f"inputs, expected dim {settings.embed_dim}")
                out.extend(vecs)
                break
            except (APIStatusError, APITimeoutError, APIConnectionError):
                if attempt == retries - 1:
                    raise
                backoff(attempt)
    return out
