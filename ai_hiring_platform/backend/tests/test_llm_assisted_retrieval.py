"""
LLM-assisted retrieval: the guarantees that must hold whatever the model does.

The feature is optional and the model is untrusted, so most of these tests are about
what happens when it is absent, slow, wrong or lying. The one rule underneath all of
them: an assisted stage may reorder or demote a REAL candidate, but it may never
invent one, resurrect an excluded one, or crash a turn.
"""
import json
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services.ai import chat_llm_retrieval as assist
from app.services.ai import chat_retrieval_service as retrieval


class FakeLLM:
    """Records prompts and replays canned responses."""

    model = "fake-model"

    def __init__(self, reply="", fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def complete(self, prompt, system="", temperature=0.0):
        self.calls += 1
        if self.fail:
            raise RuntimeError("endpoint down")
        return SimpleNamespace(text=self.reply)


@pytest.fixture(autouse=True)
def _clean():
    """Verdicts are cached across calls by design; tests must not leak into each other."""
    assist.clear_cache()
    yield
    assist.clear_cache()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_LLM_VERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_LLM_RERANK_ENABLED", True)


def _use(monkeypatch, llm):
    monkeypatch.setattr(assist.llm_service, "get_llm", lambda: llm)
    return llm


# --- Off by default ----------------------------------------------------------
def test_disabled_by_default_makes_no_calls(monkeypatch):
    """The deterministic funnel must be untouched until someone opts in."""
    # Set explicitly rather than trusting the default, so a developer with the flags
    # exported in their shell does not see this pass for the wrong reason.
    monkeypatch.setattr(settings, "RETRIEVAL_LLM_VERIFY_ENABLED", False)
    monkeypatch.setattr(settings, "RETRIEVAL_LLM_RERANK_ENABLED", False)
    llm = _use(monkeypatch, FakeLLM(reply="[]"))
    assert assist.verify_evidence([("1:0", "Java", "wrote Java services")]) == {}
    assert assist.rerank("Java", [{"resume_id": 1, "evidence": []}]) == {}
    assert llm.calls == 0


def test_enabled_but_no_model_is_silent(monkeypatch, enabled):
    monkeypatch.setattr(assist.llm_service, "get_llm", lambda: None)
    assert assist.verify_evidence([("1:0", "Java", "wrote Java")]) == {}
    assert assist.rerank("Java", [{"resume_id": 1, "evidence": []}]) == {}


# --- The model is untrusted ---------------------------------------------------
def test_verdict_for_an_id_we_never_asked_about_is_discarded(monkeypatch, enabled):
    """A model naming an unknown id is hallucinating; it must not reach the caller."""
    _use(monkeypatch, FakeLLM(reply=json.dumps([
        {"id": "1:0", "verdict": "yes"},
        {"id": "999:0", "verdict": "yes"},
    ])))
    out = assist.verify_evidence([("1:0", "Java", "wrote Java services")])
    assert out == {"1:0": "yes"}


def test_rerank_ignores_a_resume_id_it_was_not_shown(monkeypatch, enabled):
    _use(monkeypatch, FakeLLM(reply=json.dumps([
        {"id": 1, "score": 90},
        {"id": 42, "score": 99},
    ])))
    out = assist.rerank("Java", [{"resume_id": 1, "evidence": []}])
    assert out == {1: 90.0}


def test_scores_are_clamped(monkeypatch, enabled):
    _use(monkeypatch, FakeLLM(reply=json.dumps([{"id": 1, "score": 5000}])))
    assert assist.rerank("Java", [{"resume_id": 1, "evidence": []}]) == {1: 100.0}


def test_unknown_verdict_word_is_dropped(monkeypatch, enabled):
    _use(monkeypatch, FakeLLM(reply=json.dumps([{"id": "1:0", "verdict": "probably"}])))
    assert assist.verify_evidence([("1:0", "Java", "x")]) == {}


@pytest.mark.parametrize("reply", ["not json at all", "", "{]", "null"])
def test_malformed_output_yields_no_opinion(monkeypatch, enabled, reply):
    _use(monkeypatch, FakeLLM(reply=reply))
    assert assist.verify_evidence([("1:0", "Java", "x")]) == {}
    assert assist.rerank("Java", [{"resume_id": 1, "evidence": []}]) == {}


def test_json_wrapped_in_prose_is_still_read(monkeypatch, enabled):
    """Small local models fence their JSON or preface it; that must not fail a batch."""
    _use(monkeypatch, FakeLLM(reply='Sure!\n```json\n[{"id": "1:0", "verdict": "yes"}]\n```'))
    assert assist.verify_evidence([("1:0", "Java", "x")]) == {"1:0": "yes"}


def test_endpoint_failure_never_raises(monkeypatch, enabled):
    _use(monkeypatch, FakeLLM(fail=True))
    assert assist.verify_evidence([("1:0", "Java", "x")]) == {}
    assert assist.rerank("Java", [{"resume_id": 1, "evidence": []}]) == {}


# --- Caching ------------------------------------------------------------------
def test_identical_passage_is_judged_once(monkeypatch, enabled):
    llm = _use(monkeypatch, FakeLLM(reply=json.dumps([{"id": "1:0", "verdict": "yes"}])))
    assist.verify_evidence([("1:0", "Java", "wrote Java services")])
    assert llm.calls == 1
    assist.verify_evidence([("1:0", "Java", "wrote Java services")])
    assert llm.calls == 1, "a cached verdict must not cost a second round trip"


# --- The admission gate --------------------------------------------------------
def test_missing_verdict_falls_back_to_the_lexical_rule():
    """No verdict means 'not asked', never 'no'. Deterministic behaviour is preserved."""
    assert retrieval._admits({"skill": "Java", "text": "built Java microservices"}) is True
    assert retrieval._admits({"skill": "Java", "text": "designed Figma mockups"}) is False


def test_model_yes_admits_evidence_the_lexical_rule_would_reject():
    """The recall gain: demonstrated skill, word never written."""
    chunk = {"skill": "Communication",
             "text": "Ran sprint ceremonies and unblocked the team daily",
             "llm_verdict": "yes"}
    assert retrieval._confirms(chunk["text"], chunk["skill"]) is False
    assert retrieval._admits(chunk) is True


def test_model_no_rejects_evidence_the_lexical_rule_would_accept():
    """The precision gain: word present, but it is not this person's skill."""
    chunk = {"skill": "Java",
             "text": "Worked alongside the Java team while building the Python service",
             "llm_verdict": "no"}
    assert retrieval._confirms(chunk["text"], chunk["skill"]) is True
    assert retrieval._admits(chunk) is False


def test_weak_is_admitted_but_demoted_not_dropped():
    chunk = {"skill": "Java", "text": "Skills: Java, SQL", "literal": True, "llm_verdict": "weak"}
    assert retrieval._admits(chunk) is True
    assert chunk["literal"] is False, "a listed claim must score as semantic-only support"


# --- Blending into the score ----------------------------------------------------
def _candidate():
    return {
        "resume_id": 1,
        "match_percentage": 80,
        "match_parameters": [
            {"key": "skill", "label": "Skill Match", "weight": 0.6, "score": 80.0,
             "contribution": 48.0, "basis": "x"},
            {"key": "location", "label": "Location Match", "weight": 0.4, "score": 80.0,
             "contribution": 32.0, "basis": "y"},
        ],
    }


def test_relevance_appears_as_its_own_explainable_parameter():
    candidate = _candidate()
    retrieval._apply_relevance(candidate, 40.0)
    keys = [c["key"] for c in candidate["match_parameters"]]
    assert "llm_relevance" in keys
    assert sum(c["weight"] for c in candidate["match_parameters"]) == pytest.approx(1.0, abs=0.01)


def test_relevance_moves_the_score_toward_itself():
    candidate = _candidate()
    retrieval._apply_relevance(candidate, 0.0)
    assert candidate["match_percentage"] < 80, "a zero relevance judgement must cost the candidate"

    better = _candidate()
    retrieval._apply_relevance(better, 100.0)
    assert better["match_percentage"] > 80


def test_deterministic_parameters_keep_their_relative_proportions():
    """Relevance dilutes every other parameter equally — it never reweights them."""
    candidate = _candidate()
    before = candidate["match_parameters"][0]["weight"] / candidate["match_parameters"][1]["weight"]
    retrieval._apply_relevance(candidate, 50.0)
    after = candidate["match_parameters"][0]["weight"] / candidate["match_parameters"][1]["weight"]
    assert before == pytest.approx(after, abs=0.01)


def test_overview_questions_get_no_relevance_component():
    """Nothing was asked for, so there is nothing to be relevant TO."""
    candidate = {"resume_id": 1, "match_percentage": 50, "match_parameters": []}
    retrieval._apply_relevance(candidate, 90.0)
    assert candidate["match_parameters"] == []
    assert candidate["match_percentage"] == 50
