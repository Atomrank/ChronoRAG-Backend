"""extraction_mode selects exhaustive vs salient prompts."""
from app.config import settings
from app.passes import pass1_system, _pass2_system
from app.v2 import prompts


def test_prompt_version_bumped():
    assert prompts.PROMPT_VERSION.startswith("v2.2")


def test_exhaustive_prompts_avoid_major_quota(monkeypatch):
    monkeypatch.setattr(settings, "extraction_mode", "exhaustive")
    p1 = pass1_system()
    assert "EVERY narrated event" in p1
    assert "key interactions" not in p1.lower()
    p2 = _pass2_system([{"name": "main", "description": "story", "is_framing": False}])
    assert "exhaustive mode" in p2
    assert "8-15" not in p2
    assert "EXHAUSTIVE MODE" in prompts.extract_system()
    assert "Never skip an event because it seems minor" in prompts.extract_system()


def test_salient_prompts_keep_milestone_language(monkeypatch):
    monkeypatch.setattr(settings, "extraction_mode", "salient")
    p1 = pass1_system()
    assert "plot-significant" in p1
    p2 = _pass2_system([{"name": "main", "description": "story", "is_framing": False}])
    assert "salient mode" in p2
    assert "8-15" in p2
    assert "SALIENT MODE" in prompts.extract_system()
