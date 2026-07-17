"""
Tests for the provider-agnostic grounded LLM reasoning path. A fake LLM stands in
for any provider (no network), so we verify prompt grounding, robust JSON parsing,
deterministic ownership of the numbers, and graceful fallback.
"""
import json

from app.services.ai import evaluation_service, llm_service
from app.services.ai.evaluation_service import _build_reasoning_prompt, _extract_json

_EVIDENCE = {
    "analysis_id": 1,
    "retrieval_results": [
        {"requirement": "python", "importance": "must", "weight": 1.0,
         "matches": [{"chunk": "Built REST APIs in Python and FastAPI over three years.",
                      "section": "Experience", "score": 0.9, "confidence": 90}]},
        {"requirement": "kubernetes", "importance": "nice", "weight": 0.4, "matches": []},
    ],
}

# A well-formed LLM reply — with deliberately WRONG deterministic numbers (999) to
# prove the engine overrides them rather than trusting the model.
_LLM_PAYLOAD = {
    "experience_score": 80, "project_score": 40, "confidence_score": 70,
    "summary": "Strong Python backend candidate.",
    "requirements": [
        {"requirement": "python", "category": "Programming Language", "status": "Matched",
         "matched_evidence": "Built REST APIs in Python", "explanation": "Shown in Experience",
         "limitations": "None", "confidence": 88},
        {"requirement": "kubernetes", "category": "Cloud & DevOps", "status": "Missing",
         "matched_evidence": "", "explanation": "No evidence", "limitations": "Not shown",
         "confidence": 0},
    ],
    "strengths": ["Python backend depth"], "weaknesses": ["No Kubernetes"],
    "skill_relationships": [], "missing_skills": ["kubernetes"],
    "learning_roadmap": [{"skill": "kubernetes", "estimated_time": "10-14 days", "reason": "gap"}],
    "interview_questions": ["Describe a hard Python bug you fixed."],
    "recruiter_recommendation": "Proceed to technical screen",
    "rejection_email": None,
    "coverage_score": 999, "quality_score": 999, "overall_score": 999,
}


class _FakeResp:
    def __init__(self, text): self.text = text


class _FakeLLM:
    def __init__(self, text): self._t = text
    def complete(self, prompt): return _FakeResp(self._t)


def test_prompt_is_grounded_and_priority_aware():
    p = _build_reasoning_prompt(_EVIDENCE)
    assert "Reason ONLY" in p                              # grounding instruction
    assert "must-have" in p                                 # priority conveyed
    assert "Built REST APIs in Python" in p                 # real evidence embedded
    assert "kubernetes" in p


def test_extract_json_tolerates_fences_and_prose():
    assert _extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _extract_json("Sure, here it is:\n{\"a\": 2}\nHope that helps.") == {"a": 2}


def test_llm_path_overrides_deterministic_numbers(monkeypatch):
    monkeypatch.setattr(llm_service, "get_llm", lambda: _FakeLLM(json.dumps(_LLM_PAYLOAD)))
    report = evaluation_service.evaluate_evidence(_EVIDENCE)

    # LLM-owned narrative is preserved.
    assert report.summary == "Strong Python backend candidate."
    assert report.recruiter_recommendation == "Proceed to technical screen"
    # Deterministic numbers are recomputed, never the model's 999.
    assert report.coverage_score == 71   # python(1.0) matched, kubernetes(0.4) missing -> 1.0/1.4
    assert report.quality_score != 999
    assert report.overall_score != 999 and 0 <= report.overall_score <= 100
    assert report.authenticity is not None
    # Requirement importance is stamped from the evidence, not the LLM.
    py = [r for r in report.requirements if r.requirement == "python"][0]
    assert py.importance == "must"


def test_llm_failure_falls_back_to_deterministic(monkeypatch):
    class _BadLLM:
        def complete(self, prompt): return _FakeResp("this is not json")
    monkeypatch.setattr(llm_service, "get_llm", lambda: _BadLLM())
    report = evaluation_service.evaluate_evidence(_EVIDENCE)
    # Fallback still returns a valid, fully-populated report.
    assert 0 <= report.overall_score <= 100
    assert len(report.requirements) == 2
    assert report.authenticity is not None
