"""
Query-understanding and scoring tests for the recruiter chatbot.

Deterministic units only (no index, no LLM): the same question must always produce the
same search plan and the same match percentage, which is what makes a score defensible
to a candidate or an auditor.

Several cases here are regressions from real recruiter transcripts — they are the bugs
that made the assistant look unreliable, and they must never come back.
"""
import pytest

from app.services.ai import chat_retrieval_service as retrieval
from app.services.ai.chat_query_understanding import QueryKind, parse_intent
from tests.chat_fixtures import lexicon_with_skills

NAMES = ["Laura Singh", "Sneha Taylor", "John Pillai", "David Pillai", "Priya Krishnan"]
# A small real pool: understanding resolves words against the CORPUS now, because only
# the corpus can say whether a word is a technology or ordinary English.
LEXICON = lexicon_with_skills(
    NAMES,
    ["Java", "Python", "AWS", "React", "Kubernetes", "Docker", "SQL", "Machine Learning"],
)


def parse(q):
    return parse_intent(q, LEXICON)


# --- Intent classification --------------------------------------------------
def test_parses_skill_and_minimum_years():
    intent = parse("need candidate with 10 yr experience in java")
    assert intent.kind is QueryKind.SKILL_SEARCH
    assert intent.min_years == 10.0
    assert intent.skills == ["Java"]


@pytest.mark.parametrize("phrasing", [
    "at least 5 years of python",
    "5+ years python",
    "minimum 5 years in python",
])
def test_years_phrasings(phrasing):
    assert parse(phrasing).min_years == 5.0


# --- REGRESSION: request scaffolding must never become a skill --------------
def test_let_me_know_more_about_is_a_profile_request_not_a_skill_search():
    """'let me know more about David Pillai' once searched for the skill "let"."""
    intent = parse("let me know more about David Pillai")
    assert intent.kind is QueryKind.PROFILE
    assert intent.named_candidates == ["David Pillai"]
    assert intent.skills == []


@pytest.mark.parametrize("phrasing", [
    "tell me more about Priya Krishnan",
    "who is Priya Krishnan",
    "give me details about Priya Krishnan",
    "I want an overview of Priya Krishnan",
])
def test_profile_phrasings_never_extract_skills(phrasing):
    intent = parse(phrasing)
    assert intent.kind is QueryKind.PROFILE
    assert intent.skills == []


def test_ordinary_english_words_are_never_skills():
    """'who is the father of the nation' once produced skills [father, nation]."""
    intent = parse("who is the father of the nation")
    assert intent.skills == []
    assert intent.kind is QueryKind.AMBIGUOUS
    assert intent.clarification


def test_candidate_name_is_not_treated_as_a_skill():
    intent = parse("how much java does Laura Singh have?")
    assert intent.named_candidates == ["Laura Singh"]
    assert intent.skills == ["Java"]
    assert not {"laura", "singh"} & {s.lower() for s in intent.skills}


# --- REGRESSION: typos are corrected, not searched verbatim -----------------
@pytest.mark.parametrize("typed,expected", [
    ("javaa", "Java"),
    ("pythonn", "Python"),
    ("kubernets", "Kubernetes"),
    ("dockr", "Docker"),
])
def test_typos_are_corrected_to_real_skills(typed, expected):
    """
    Casing comes from the shared technology taxonomy, not from the caller's vocabulary:
    corrections resolve against known technologies ONLY, so a stray corpus token can
    never be the correction target (that is what turned "aravind" into "AAVIN").
    """
    intent = parse(f"who knows {typed}")
    assert [s.lower() for s in intent.skills] == [expected.lower()]
    assert [(t, c.lower()) for t, c in intent.corrections] == [(typed, expected.lower())]


def test_correct_spelling_reports_no_correction():
    assert parse("who knows java").corrections == []


# --- REGRESSION: unknown people produce a question, not a guess -------------
def test_unknown_name_is_flagged_for_clarification():
    """'is there anyone named nithin?' once answered with an unrelated candidate."""
    intent = parse("is there anyone named nithin?")
    assert intent.kind is QueryKind.EXISTENCE
    assert intent.named_candidates == []
    assert intent.unresolved_name
    assert "nithin" in intent.unresolved_name.lower()


def test_existence_query_for_a_real_person_resolves():
    intent = parse("is there anyone named Laura Singh?")
    assert intent.named_candidates == ["Laura Singh"]


# --- Scoring: depth, not a found/not-found flag -----------------------------
def _evidence(skill, section, sim=0.75, literal=True):
    return {"skill": skill, "text": f"{skill} work", "section": section,
            "similarity": sim, "literal": literal}


def test_skill_depth_ranks_sections_correctly():
    skills_only, _ = retrieval._skill_depth([_evidence("Java", "Skills")])
    experience, _ = retrieval._skill_depth([_evidence("Java", "Experience")])
    everywhere, _ = retrieval._skill_depth([
        _evidence("Java", "Skills"), _evidence("Java", "Experience"),
        _evidence("Java", "Projects"), _evidence("Java", "Certifications"),
    ])
    assert skills_only < experience < everywhere
    assert skills_only < 0.5, "a bare Skills-list mention must not read as proven"
    assert everywhere >= 0.99, "proof across every section should be full depth"


def test_coverage_varies_instead_of_always_being_100():
    """Coverage was a flat 100% whenever the word appeared anywhere."""
    intent = parse("java developer")
    profile = {"resume_id": 1, "name": "A", "total_years": 6.0}

    listed = retrieval._score_candidate(profile, [_evidence("Java", "Skills")], intent)
    proven = retrieval._score_candidate(
        profile,
        [_evidence("Java", "Skills"), _evidence("Java", "Experience"),
         _evidence("Java", "Projects"), _evidence("Java", "Certifications")],
        intent,
    )
    assert listed["breakdown"]["skill_coverage"] < 50
    assert proven["breakdown"]["skill_coverage"] > 90
    assert proven["match_percentage"] > listed["match_percentage"]


def test_demonstrated_evidence_outranks_a_bare_skills_list():
    intent = parse("java developer with 5 years")
    profile = {"resume_id": 1, "name": "A", "total_years": 6.0}
    demonstrated = retrieval._score_candidate(profile, [_evidence("Java", "Experience")], intent)
    listed = retrieval._score_candidate({**profile, "resume_id": 2}, [_evidence("Java", "Skills")], intent)
    assert demonstrated["match_percentage"] > listed["match_percentage"]
    assert demonstrated["demonstrated_skills"] == ["Java"]
    assert listed["listed_only_skills"] == ["Java"]


def test_semantic_only_evidence_scores_below_literal():
    intent = parse("java developer")
    profile = {"resume_id": 1, "name": "A", "total_years": 6.0}
    literal = retrieval._score_candidate(profile, [_evidence("Java", "Experience", literal=True)], intent)
    semantic = retrieval._score_candidate(profile, [_evidence("Java", "Experience", literal=False)], intent)
    assert semantic["breakdown"]["skill_coverage"] < literal["breakdown"]["skill_coverage"]


def test_missing_skills_are_reported_not_invented():
    intent = parse("java and kubernetes developer")
    scored = retrieval._score_candidate(
        {"resume_id": 1, "name": "A", "total_years": 8.0},
        [_evidence("Java", "Experience")],
        intent,
    )
    assert scored["missing_skills"] == ["Kubernetes"]
    assert scored["skill_depth"]["Kubernetes"] == 0


def test_experience_fit_scoring():
    assert retrieval._experience_fit(12.0, 10.0)[0] > 0.9
    assert retrieval._experience_fit(10.0, 10.0)[0] >= 0.9
    assert retrieval._experience_fit(4.0, 10.0)[0] < 0.5
    assert retrieval._experience_fit(None, 10.0)[0] == 0.5
    assert retrieval._experience_fit(3.0, None)[0] == 1.0


def test_scoring_is_reproducible():
    intent = parse("python developer 5 years")
    profile = {"resume_id": 1, "name": "A", "total_years": 7.0}
    ev = [_evidence("Python", "Experience", 0.71)]
    assert (retrieval._score_candidate(profile, ev, intent)["match_percentage"]
            == retrieval._score_candidate(profile, ev, intent)["match_percentage"])
