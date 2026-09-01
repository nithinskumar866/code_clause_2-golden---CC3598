"""
Prompt Lab tests.

The grading engine is the part that must never drift: it is what tells a recruiter that
a prompt rewrite genuinely fixed the link policy rather than merely reading better. So
every rule is asserted in both directions — a compliant answer passes, a violating
answer fails, and an answer the rule has nothing to say about is NA rather than a free
pass.

No test here calls an LLM. Generation is stubbed where a run is exercised end to end,
which keeps the suite deterministic and green with no key configured (Golden Rule 10).
"""
import json

import pytest

from app.services.ai import prompt_lab_service as lab
from app.services.ai import prompt_rules_service as rules
from app.services.ai.prompt_rules_service import Expectations, Turn

PROFILE = (
    "PEOPLE DIRECTORY:\n"
    "Name=Priya Raman | Role=Senior Data Engineer | Team=Platform | "
    "LinkedIn=https://www.linkedin.com/in/priya-raman | "
    "Skills=Apache Spark, Airflow, dbt"
)

THREAD = [
    {"role": "user", "content": "Who is Priya Raman?"},
    {"role": "assistant", "content":
        "Priya Raman is a Senior Data Engineer on Platform. https://www.linkedin.com/in/priya-raman"},
]


def grade(answer, exp=None, context=PROFILE, question="Who is Priya Raman?", history=None, prompt=""):
    return rules.evaluate(
        answer=answer,
        turn=Turn(question=question, context=context, history=history or []),
        expectations=Expectations.from_dict(exp or {}),
        prompt=prompt,
    )


def verdict(result, rule_id):
    return next(r["status"] for r in result["rules"] if r["rule"] == rule_id)


# ---------------------------------------------------------------------------- shape


def test_answer_length_passes_within_limit_and_fails_beyond_it():
    short = "Priya Raman is a Senior Data Engineer on the Platform team."
    long = "\n".join(["Priya Raman is a Senior Data Engineer.", "She works on Platform.", "She is helpful."])
    assert verdict(grade(short, {"max_lines": 2}), "answer_length") == rules.PASS
    assert verdict(grade(long, {"max_lines": 2}), "answer_length") == rules.FAIL


def test_blank_lines_do_not_count_towards_the_line_limit():
    answer = "Priya Raman is a Senior Data Engineer.\n\n\nShe is on the Platform team."
    assert verdict(grade(answer, {"max_lines": 2}), "answer_length") == rules.PASS


def test_brevity_catches_one_very_long_line_that_the_line_rule_misses():
    answer = "Priya Raman is a Senior Data Engineer. " * 20
    result = grade(answer, {"max_lines": 2, "max_chars": 200})
    assert verdict(result, "answer_length") == rules.PASS
    assert verdict(result, "answer_brevity") == rules.FAIL


# ----------------------------------------------------------------------------- links


def test_markdown_link_fails_and_a_plain_url_passes():
    md = "Priya Raman is a Senior Data Engineer. [profile](https://www.linkedin.com/in/priya-raman)"
    plain = "Priya Raman is a Senior Data Engineer. https://www.linkedin.com/in/priya-raman"
    assert verdict(grade(md, {"link_allowed": True}), "plain_urls") == rules.FAIL
    assert verdict(grade(plain, {"link_allowed": True}), "plain_urls") == rules.PASS


def test_plain_url_rule_is_not_applicable_when_there_is_no_url():
    assert verdict(grade("Priya Raman is a Senior Data Engineer."), "plain_urls") == rules.NA


def test_link_given_where_the_turn_forbids_one_fails():
    answer = "She is on the Platform team. https://www.linkedin.com/in/priya-raman"
    assert verdict(grade(answer, {"link_allowed": False}), "link_policy") == rules.FAIL


def test_link_missing_where_the_turn_requires_one_fails():
    assert verdict(grade("She is on the Platform team.", {"link_required": True}), "link_policy") == rules.FAIL


def test_repeating_a_link_already_given_in_the_thread_fails():
    answer = "She is on the Platform team. https://www.linkedin.com/in/priya-raman"
    result = grade(answer, {"link_allowed": True}, history=THREAD, question="Which team is he on again?")
    assert verdict(result, "no_repeated_link") == rules.FAIL


def test_repeat_rule_stands_down_when_the_user_asks_for_the_link():
    answer = "https://www.linkedin.com/in/priya-raman"
    result = grade(answer, {"link_required": True, "link_allowed": True}, history=THREAD,
                   question="Send me her profile link")
    assert verdict(result, "no_repeated_link") == rules.NA
    assert verdict(result, "link_policy") == rules.PASS


def test_a_url_absent_from_the_context_is_ungrounded():
    answer = "Priya Raman is a Senior Data Engineer. https://example.com/priya"
    assert verdict(grade(answer, {"link_allowed": True}), "grounded_urls") == rules.FAIL


def test_the_derived_skills_url_counts_as_grounded():
    answer = "Spark, Airflow and dbt. https://www.linkedin.com/in/priya-raman/details/skills/"
    result = grade(answer, {"skills_requested": True, "link_allowed": True}, question="What are her skills?")
    assert verdict(result, "grounded_urls") == rules.PASS
    assert verdict(result, "skills_url") == rules.PASS


def test_a_skills_answer_without_the_skills_url_fails():
    result = grade("Spark, Airflow and dbt.", {"skills_requested": True}, question="What are her skills?")
    assert verdict(result, "skills_url") == rules.FAIL


def test_an_explicit_linkedin_skills_url_in_context_is_the_one_expected():
    context = PROFILE + "\nLinkedInSkills=https://www.linkedin.com/in/priya-raman/details/skills/"
    assert rules.skills_url_for(context) == "https://www.linkedin.com/in/priya-raman/details/skills/"


# ----------------------------------------------------------------------------- years


def test_years_asserted_without_context_support_fails():
    answer = "She has around 8 years of experience."
    assert verdict(grade(answer, {"years_known": False}), "years_policy") == rules.FAIL


def test_declining_to_state_unverified_years_passes():
    answer = "Exact years are not verified in the current data."
    assert verdict(grade(answer, {"years_known": False}), "years_policy") == rules.PASS


def test_years_present_in_context_must_be_stated_and_must_match():
    context = PROFILE + " | Experience=9 years"
    exp = {"years_known": True, "expected_years": "9"}
    right = grade("She has 9 years in data engineering.", exp, context=context)
    wrong = grade("She has 6 years in data engineering.", exp, context=context)
    silent = grade("She is a Senior Data Engineer.", exp, context=context)
    assert verdict(right, "years_policy") == rules.PASS
    assert verdict(wrong, "years_policy") == rules.FAIL
    assert verdict(silent, "years_policy") == rules.FAIL


def test_a_number_absent_from_the_context_is_ungrounded():
    assert verdict(grade("She manages 14 pipelines."), "grounded_numbers") == rules.FAIL


def test_numbers_present_in_the_context_are_grounded():
    context = PROFILE + " | Experience=9 years"
    assert verdict(grade("She has 9 years.", {"years_known": True}, context=context), "grounded_numbers") == rules.PASS


# ---------------------------------------------------------------------------- shape 2


def test_an_unrequested_list_fails_but_a_requested_one_does_not():
    answer = "Roles you might like:\n- Data Engineer\n- Analytics Engineer"
    assert verdict(grade(answer, {"jobs_requested": False}), "no_unrequested_list") == rules.FAIL
    assert verdict(grade(answer, {"jobs_requested": True}), "no_unrequested_list") == rules.NA


def test_a_single_bullet_is_not_treated_as_a_list():
    assert verdict(grade("She is on Platform.\n- Spark"), "no_unrequested_list") == rules.PASS


def test_skills_answer_must_name_skills_from_the_context():
    good = grade("Spark, Airflow and dbt.", {"skills_requested": True}, question="skills?")
    bad = grade("She is skilled in many areas.", {"skills_requested": True}, question="skills?")
    assert verdict(good, "skills_listed") == rules.PASS
    assert verdict(bad, "skills_listed") == rules.FAIL


def test_skills_rules_are_not_applicable_to_a_non_skills_question():
    result = grade("She is a Senior Data Engineer.", {"skills_requested": False})
    assert verdict(result, "skills_listed") == rules.NA
    assert verdict(result, "skills_url") == rules.NA


def test_reproducing_a_prompt_line_is_an_instruction_leak():
    prompt = "- Keep the answer to 1-2 short lines about who the person is at the company."
    leaked = "Keep the answer to 1-2 short lines about who the person is at the company. She is on Platform."
    assert verdict(grade(leaked, prompt=prompt), "no_instruction_leak") == rules.FAIL
    assert verdict(grade("She is on Platform.", prompt=prompt), "no_instruction_leak") == rules.PASS


def test_case_specific_required_and_forbidden_strings():
    result = grade("She is a Senior Data Engineer.", {
        "must_contain": ["Senior Data Engineer"], "must_not_contain": ["as an AI"],
    })
    assert verdict(result, "must_contain") == rules.PASS
    assert verdict(result, "must_not_contain") == rules.PASS
    bad = grade("As an AI, I cannot say.", {"must_contain": ["Senior Data Engineer"]})
    assert verdict(bad, "must_contain") == rules.FAIL


# ---------------------------------------------------------------------------- scoring


def test_not_applicable_rules_never_count_as_passes():
    result = grade("Priya Raman is a Senior Data Engineer on Platform.")
    applicable = result["passed"] + result["failed"]
    assert result["not_applicable"] > 0
    assert applicable + result["not_applicable"] == len(result["rules"])
    assert result["score"] == pytest.approx(100.0 * result["passed"] / applicable)


def test_the_starter_suite_covers_every_clause_it_claims_to():
    suite = lab.starter_suite()
    assert suite["prompt"].strip()
    ids = {c["id"] for c in suite["cases"]}
    assert {"intro", "years-unknown", "years-known", "skills", "pronoun-followup",
            "no-job-list", "explicit-link", "unknown-person"} <= ids


def test_a_compliant_answer_to_the_starter_intro_case_scores_full_marks():
    case = next(c for c in lab.starter_suite()["cases"] if c["id"] == "intro")
    answer = ("Priya Raman is a Senior Data Engineer on the Platform team. "
              "https://www.linkedin.com/in/priya-raman")
    result = lab.score_case(lab.STARTER_PROMPT, case, answer)
    assert result["failed"] == 0, result["violations"]
    assert result["score"] == 100.0


def test_the_violating_answer_the_prompt_is_accused_of_is_caught():
    """The reported failure mode: long, markdown-linked, invented years, unasked job list."""
    case = next(c for c in lab.starter_suite()["cases"] if c["id"] == "years-unknown")
    answer = (
        "Priya Raman has approximately 12 years of experience in data engineering.\n"
        "You can view her profile here: [LinkedIn](https://www.linkedin.com/in/priya-raman)\n"
        "Roles that may suit her:\n- Principal Data Engineer\n- Head of Platform"
    )
    result = lab.score_case(lab.STARTER_PROMPT, case, answer)
    caught = {r["rule"] for r in result["rules"] if r["status"] == rules.FAIL}
    assert {"answer_length", "plain_urls", "years_policy",
            "grounded_numbers", "no_unrequested_list"} <= caught


# ------------------------------------------------------------------------ run + API


def _stub_llm(monkeypatch, answer):
    monkeypatch.setattr(lab, "generate", lambda prompt, turn, temperature=0.2: {
        "answer": answer, "error": None, "latency_ms": 1,
    })
    monkeypatch.setattr(lab.llm_service, "get_llm", lambda: object())
    monkeypatch.setattr(lab, "_model_label", lambda: "stub-model")


def test_run_variant_aggregates_per_rule(monkeypatch):
    _stub_llm(monkeypatch, "Priya Raman is a Senior Data Engineer on the Platform team.")
    result = lab.run_variant(lab.STARTER_PROMPT, lab.starter_suite()["cases"], label="A")
    assert result["total_cases"] == 8
    assert result["score"] is not None
    assert {r["rule"] for r in result["by_rule"]} <= {r["id"] for r in rules.RULE_CATALOG}


def test_compare_reports_a_delta_per_rule(monkeypatch):
    _stub_llm(monkeypatch, "Priya Raman is a Senior Data Engineer on the Platform team.")
    result = lab.compare(
        [{"label": "A", "prompt": lab.STARTER_PROMPT}, {"label": "B", "prompt": "Be brief."}],
        lab.starter_suite()["cases"],
    )
    assert len(result["variants"]) == 2
    assert result["deltas"] and all("delta" in d for d in result["deltas"])


def test_generate_reports_a_missing_llm_rather_than_inventing_an_answer(monkeypatch):
    monkeypatch.setattr(lab.llm_service, "get_llm", lambda: None)
    out = lab.generate("p", Turn(question="q"))
    assert out["answer"] == ""
    assert "No LLM" in out["error"]


def test_rules_endpoint_lists_the_catalog(client):
    res = client.get("/api/v1/prompt-lab/rules")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] and len(body["data"]) == len(rules.RULE_CATALOG)


def test_starter_endpoint_serves_the_prompt_and_its_cases(client):
    res = client.get("/api/v1/prompt-lab/starter")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["prompt"].strip() and len(data["cases"]) == 8


def test_score_endpoint_grades_without_an_llm(client):
    res = client.post("/api/v1/prompt-lab/score", json={
        "prompt": lab.STARTER_PROMPT,
        "case": {"question": "Who is Priya Raman?", "context": PROFILE,
                 "expectations": {"max_lines": 2, "link_allowed": True}},
        "answer": "Priya Raman is a Senior Data Engineer on Platform. https://www.linkedin.com/in/priya-raman",
    })
    assert res.status_code == 200
    assert res.json()["data"]["failed"] == 0


def test_run_endpoint_without_an_llm_says_so_and_stores_nothing(client, monkeypatch):
    # get_llm is stubbed rather than left to the ambient config: conftest's
    # isolate_llm_config blanks the provider KEYS but not LLM_RUNTIME/RUNPOD_BASE_URL,
    # so on a machine whose .env names a self-hosted runtime this test would otherwise
    # build a real client and call out over the network.
    monkeypatch.setattr(lab.llm_service, "get_llm", lambda: None)

    res = client.post("/api/v1/prompt-lab/run", json={
        "variants": [{"label": "A", "prompt": lab.STARTER_PROMPT}],
        "cases": [{"question": "Who is Priya Raman?", "context": PROFILE}],
    })
    assert res.status_code == 200
    body = res.json()
    assert body["data"]["llm_available"] is False
    assert client.get("/api/v1/prompt-lab/runs").json()["data"] == []


def test_suite_round_trip_and_deletion(client):
    created = client.post("/api/v1/prompt-lab/suites", json={
        "name": "Person query", "description": "field prompt",
        "prompt": lab.STARTER_PROMPT,
        "cases": [{"question": "Who is Priya Raman?", "context": PROFILE}],
    }).json()["data"]
    assert created["id"] and len(created["cases"]) == 1

    updated = client.put(f"/api/v1/prompt-lab/suites/{created['id']}", json={
        "prompt": "Shorter prompt.",
    }).json()["data"]
    assert updated["prompt"] == "Shorter prompt."

    listed = client.get("/api/v1/prompt-lab/suites").json()["data"]
    assert [s["id"] for s in listed] == [created["id"]]

    assert client.delete(f"/api/v1/prompt-lab/suites/{created['id']}").status_code == 200
    assert client.get("/api/v1/prompt-lab/suites").json()["data"] == []


def test_unknown_suite_and_run_are_404(client):
    assert client.put("/api/v1/prompt-lab/suites/9999", json={"name": "x"}).status_code == 404
    assert client.delete("/api/v1/prompt-lab/suites/9999").status_code == 404
    assert client.get("/api/v1/prompt-lab/runs/9999").status_code == 404


def test_a_run_is_recorded_in_the_regression_history(client, monkeypatch):
    from app.api.v1.routers import prompt_lab as router_module

    _stub_llm(monkeypatch, "Priya Raman is a Senior Data Engineer on the Platform team.")
    monkeypatch.setattr(router_module.lab.llm_service, "get_llm", lambda: object())

    suite = client.post("/api/v1/prompt-lab/suites", json={
        "name": "Person query", "prompt": lab.STARTER_PROMPT,
        "cases": [{"question": "Who is Priya Raman?", "context": PROFILE}],
    }).json()["data"]

    run = client.post("/api/v1/prompt-lab/run", json={
        "variants": [{"label": "A", "prompt": lab.STARTER_PROMPT}],
        "cases": [{"question": "Who is Priya Raman?", "context": PROFILE,
                   "expectations": {"max_lines": 2}}],
        "suite_id": suite["id"],
    }).json()
    assert run["data"]["llm_available"] is True

    history = client.get(f"/api/v1/prompt-lab/runs?suite_id={suite['id']}").json()["data"]
    assert len(history) == 1 and history[0]["model"] == "stub-model"

    detail = client.get(f"/api/v1/prompt-lab/runs/{history[0]['id']}").json()["data"]
    assert detail["results"]["cases"][0]["answer"].startswith("Priya Raman")
    assert json.loads(json.dumps(detail["results"]))  # round-trips as JSON


def test_a_run_where_the_endpoint_failed_is_not_written_to_history(client, monkeypatch):
    """A configured-but-dead endpoint scores empty strings. That is a fault in the
    infrastructure, not a regression in the prompt, and recording it would put a
    phantom drop in the history."""
    monkeypatch.setattr(lab, "generate", lambda prompt, turn, temperature=0.2: {
        "answer": "", "error": "Connection refused", "latency_ms": 5,
    })
    monkeypatch.setattr(lab.llm_service, "get_llm", lambda: object())
    monkeypatch.setattr(lab, "_model_label", lambda: "dead-endpoint")

    body = client.post("/api/v1/prompt-lab/run", json={
        "variants": [{"label": "A", "prompt": lab.STARTER_PROMPT}],
        "cases": [{"question": "Who is Priya Raman?", "context": PROFILE}],
    }).json()

    assert body["data"]["llm_available"] is True
    assert "returned nothing" in body["message"]
    assert client.get("/api/v1/prompt-lab/runs").json()["data"] == []
