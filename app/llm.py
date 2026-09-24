import copy
import json
import random
import re
import threading
import time
from typing import Any

from openai import AzureOpenAI, APIStatusError, APITimeoutError, APIConnectionError, BadRequestError

from .config import settings

_client: AzureOpenAI | None = None
_lock = threading.Lock()

# Cumulative token accounting, per-thread-safe
_usage = {"prompt": 0, "completion": 0}


class ContentFilterError(RuntimeError):
    """Azure OpenAI blocked the prompt/response under its content policy."""


class InputTooLongError(ValueError):
    """Input exceeds settings.llm_max_input_chars. Callers must window the text;
    it is never cut silently."""


# Build-health events (content-filter hits, degraded retries). Callers drain
# these and persist them to build_events so data loss is always visible.
_events: list[dict] = []


def _log_event(kind: str, **detail) -> None:
    with _lock:
        _events.append({"kind": kind, **detail})


def drain_events() -> list[dict]:
    with _lock:
        out = list(_events)
        _events.clear()
    return out


def _is_content_filter(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "content_filter" in text or "content management policy" in text:
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
        _client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=240.0,
            max_retries=0,          # we do our own backoff
        )
    return _client


_tls = threading.local()


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
         degrade_on_filter: bool = False) -> str:
    """degrade_on_filter=False (default): a content-filter block raises
    ContentFilterError after one retry instead of silently shrinking the input.
    Only interactive query paths should pass True."""
    payloads = [(system, _SAFE_PREFIX + _sanitize_for_azure(user))]
    if degrade_on_filter:
        payloads += [
            (system, _SAFE_PREFIX + _sanitize_for_azure(user, limit=4500)),
            (
                "You answer brief educational questions about classic literature using only the notes given.",
                _SAFE_PREFIX + _sanitize_for_azure(user, limit=2000),
            ),
        ]
    last_exc: BaseException | None = None
    for attempt in range(retries):
        sys_msg, user_msg = payloads[min(attempt, len(payloads) - 1)]
        try:
            resp = client().chat.completions.create(
                model=settings.azure_chat_deployment,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _record(resp.usage)
            return (resp.choices[0].message.content or "").strip()
        except BadRequestError as exc:
            last_exc = exc
            if _is_content_filter(exc):
                _log_event("content_filter", attempt=attempt,
                           degraded=attempt > 0 and degrade_on_filter,
                           input_chars=len(user_msg))
                if attempt == retries - 1 or not degrade_on_filter and attempt >= 1:
                    raise ContentFilterError(
                        "Azure content filter blocked this literary prompt. "
                        "In Azure AI Foundry → your gpt-4o deployment → Content filter, "
                        "set Hate/Violence to Annotate or lowest block level, then retry."
                    ) from exc
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
                    deployment: str | None = None, max_input_chars: int | None = None):
    """Return an instance of model_cls, guaranteed to validate."""
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
    last_exc: BaseException | None = None
    for attempt in range(retries):
        sys_msg, user_msg = payloads[min(attempt, len(payloads) - 1)]
        try:
            resp = client().chat.completions.create(
                model=deployment or settings.azure_chat_deployment,
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
            raw = resp.choices[0].message.content
            if not raw:
                raise ValueError("empty structured response")
            return model_cls.model_validate_json(raw)
        except BadRequestError as exc:
            last_exc = exc
            if _is_content_filter(exc):
                _log_event("content_filter", attempt=attempt,
                           degraded=attempt > 0 and degrade_on_filter,
                           input_chars=len(user_msg))
                if attempt == retries - 1 or not degrade_on_filter and attempt >= 1:
                    raise ContentFilterError(
                        "Azure content filter blocked this literary prompt. "
                        "In Azure AI Foundry → your gpt-4o deployment → Content filter, "
                        "set Hate/Violence to Annotate or lowest block level, then retry."
                    ) from exc
                backoff(min(attempt, 1))
                continue
            if attempt == retries - 1:
                raise
            backoff(attempt)
        except (APIStatusError, APITimeoutError, APIConnectionError, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            backoff(attempt)
    raise RuntimeError("unreachable")


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------
def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Batch-embed. Azure caps a single request; we chunk at 96 inputs."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 96):
        window = [t.replace("\n", " ")[:8000] for t in texts[i:i + 96]]
        for attempt in range(retries):
            try:
                resp = client().embeddings.create(
                    model=settings.azure_embed_deployment,
                    input=window,
                )
                out.extend([d.embedding for d in resp.data])
                break
            except (APIStatusError, APITimeoutError, APIConnectionError):
                if attempt == retries - 1:
                    raise
                backoff(attempt)
    return out
