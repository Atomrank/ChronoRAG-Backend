"""Text scrubbing for eval / LLM surfaces (no book-specific content)."""
from __future__ import annotations

import re

# Gold event ids embedded in synth PDF text / legacy question strings.
E_TAG_RE = re.compile(r"\s*\[E\d{4}\]")

# Discourse / connective prefixes that must not appear in question phrases.
_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:"
    r"Meanwhile,\s+in another part of [^,]+,\s+"
    r"|At the same hour,\s+"
    r"|Afterwards,\s+"
    r"|After that,\s+"
    r"|Later,\s+"
    r"|Then,\s+"
    r"|Next,\s+"
    r"|Before that,\s+"
    r"|Earlier,\s+"
    r"|Prior to this,\s+"
    r"|First,\s+"
    r")",
    re.I,
)


def strip_event_ids(text: str) -> str:
    """Remove [E####] tags. IDs belong only in gold annotations, never in LLM prompts."""
    return E_TAG_RE.sub("", text or "")


def scrub_llm_text(text: str) -> str:
    """Normalize text before it is sent to a generate/answer/ground LLM."""
    t = strip_event_ids(text)
    t = re.sub(r"\s+\.", ".", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def strip_discourse_prefix(text: str) -> str:
    t = (text or "").strip()
    while True:
        n = _DISCOURSE_PREFIX_RE.sub("", t, count=1)
        if n == t:
            break
        t = n.strip()
    return t
