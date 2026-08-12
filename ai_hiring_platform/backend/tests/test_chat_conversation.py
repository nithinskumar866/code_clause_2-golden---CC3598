"""
Conversational behaviour tests — every case is a real transcript failure.

The assistant must hold a subject across turns ("explain his skills"), refuse to act
as an encyclopedia ("what is java" used to list Java developers), and let a recruiter
ask about a person the way people actually type ("how is priya").
"""
import pytest

from app.services.ai import chat_guardrails as guards
from app.services.ai import chat_service
from app.services.ai.chat_lexicon import Person, name_tokens
from app.services.ai.chat_query_understanding import QueryKind, parse_intent

from tests.chat_fixtures import lexicon_with_skills

NAMES = ["Priya Sharma", "Arjun Mehta", "David Pillai"]
LEXICON = lexicon_with_skills(NAMES, ["Java", "Python", "Kubernetes", "AWS"])


def parse(q):
    return parse_intent(q, LEXICON)


# --- REGRESSION: definition questions are not candidate searches ------------
@pytest.mark.parametrize("q", [
    "what is java",
    "what is kubernetes",
    "define python",
    "explain kubernetes",
    "how does java work",
    "why should i use python",
])
def test_definition_questions_are_refused_not_searched(q):
    """'what is java' answered by ranking Java developers is not an answer."""
    assert parse(q).kind is QueryKind.DEFINITION


@pytest.mark.parametrize("q", [
    "who has java experience",
    "need a java developer with 10 years",
    "which candidates know kubernetes",
])
def test_real_skill_searches_are_not_mistaken_for_definitions(q):
    assert parse(q).kind is QueryKind.SKILL_SEARCH


def test_question_about_a_named_person_is_not_a_definition():
    """'what is Priya Sharma' asks about a person, not a technology."""
    assert parse("what is Priya Sharma").kind is QueryKind.PROFILE


# --- REGRESSION: first-name questions were refused as out of scope ----------
@pytest.mark.parametrize("q", ["how is priya", "say abt priya", "what is priya",
                               "tell me about arjun"])
def test_first_name_questions_reach_the_assistant(q):
    """The guardrail matched full names only, so these were wrongly blocked."""
    assert guards.check_query(q, known_names=NAMES).allowed


def test_trivia_is_still_blocked_even_though_names_are_matched_loosely():
    assert not guards.check_query("who is the father of the nation", known_names=NAMES).allowed


# --- REGRESSION: the assistant must remember who "he"/"she" is -------------
def _person(resume_id, name):
    return Person(resume_id=resume_id, name=name, tokens=name_tokens(name))


def _session(subject=None, previous=None):
    return {"turns": [], "last_candidates": previous or [], "last_intent": None,
            "subject": subject, "updated": 0}


def test_pronoun_resolves_to_the_candidate_under_discussion():
    """'explain his skills' after asking about Arjun must not reply 'who?'."""
    arjun = {"resume_id": 3, "name": "Arjun Mehta"}
    session = _session(
        subject=_person(3, "Arjun Mehta"),
        previous=[arjun, {"resume_id": 9, "name": "Someone Else"}],
    )
    for follow_up in ["explain his skills", "what about her education",
                      "tell me more about them", "what are his projects"]:
        assert chat_service._resolve_reference(follow_up, session) == arjun


def test_pronoun_without_a_subject_resolves_to_nothing():
    assert chat_service._resolve_reference("explain his skills", _session()) is None


def test_ordinal_still_wins_over_the_current_subject():
    a = {"resume_id": 1, "name": "A"}
    b = {"resume_id": 2, "name": "B"}
    session = _session(subject=_person(1, "A"), previous=[a, b])
    assert chat_service._resolve_reference("tell me more about the 2nd one", session) == b


def test_a_pronoun_is_answerable_once_a_subject_exists():
    """
    REGRESSION: "where is he from" was refused as off-topic. Scope is a property of the
    CONVERSATION — after "tell me about Naveen" that question is entirely in scope.
    """
    assert not guards.check_query("where is he from", NAMES).allowed
    assert guards.check_query("where is he from", NAMES, conversation_has_subject=True).allowed


def test_a_new_name_is_not_overridden_by_memory():
    """Naming someone explicitly must switch the subject, not stick to the old one."""
    intent = parse("tell me about David Pillai")
    assert intent.named_candidates == ["David Pillai"]


# --- Junk tokens must never reach recruiter-facing copy --------------------
def test_only_real_technologies_are_named_to_the_recruiter():
    """Extraction casts a wide net; the UI must not repeat '5M, hubs, week'."""
    noisy = ["5M", "hubs", "week", "US", "python", "java", "reviews", "aws"]
    assert chat_service._recognised_skills(noisy) == ["python", "java", "aws"]
