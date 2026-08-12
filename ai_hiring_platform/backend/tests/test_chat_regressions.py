"""
End-to-end regressions from a real recruiter transcript.

Every test here is a question a recruiter actually asked and an answer they rightly
called nonsense. They run through the WHOLE pipeline — guardrails, understanding,
retrieval, scoring, narration — against a small but real corpus, because each failure
was a interaction between layers rather than a bug inside one of them:

  * "best candidate in t nagar who knows java" returned a candidate from Manjunath
    Nagar with Ansible, because the literal word "candidate" matched the placeholder
    NAMES ("Candidate #124") given to resumes whose name extraction failed, and the
    resulting "person lookup" discarded the Java requirement entirely.
  * "who has now experience" searched for the skill "Now".
  * "support your point ..." re-ran retrieval, read "Point" as a second required
    skill, and reported 56% for the candidate it had just scored at 88%.
  * "where is he from" was refused as off-topic.
  * "where is naveen k from" found nobody, because the pool stores "K Naveen".

The LLM is never involved (`use_llm=False`): these are properties of the deterministic
engine, so they hold whether or not the model endpoint is up.
"""
from __future__ import annotations

import re
import zlib
from typing import Any, Dict, List

import faiss
import numpy as np
import pytest

from app.services.ai import chat_service, embedding_store
from app.services.ai.embedding_store import ModelIndex

_DIM = 96
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.-]*")


class _HashingEngine:
    """
    A deterministic stand-in for the embedding model.

    Tests must assert on ROUTING and FACET logic, which real embeddings would make
    non-reproducible and slow (a model download per CI run). Hashing each token into a
    fixed dimension keeps cosine similarity correlated with genuine word overlap, so
    retrieval still behaves like retrieval — it simply has no semantic generalisation,
    which none of these regressions depend on.

    CRC32 rather than `hash()`: Python salts string hashing per process, so `hash()`
    would give a different score for the same resume on every run, and a scoring
    regression could pass or fail by luck.
    """

    name = "test"
    dimension = _DIM
    min_similarity = 0.35

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(_DIM, dtype="float32")
        for token in _TOKEN.findall((text or "").lower()):
            vec[zlib.crc32(token.encode()) % _DIM] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text).reshape(1, _DIM)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.asarray([self._vector(t) for t in texts], dtype="float32")


# --- The pool -----------------------------------------------------------------
# Deliberately mirrors the real corpus's messiness: two people share a first name, one
# resume's name could not be extracted (so it carries a placeholder), and the ordinary
# English words that used to become skills are present in the body text.
_PROSE = (
    "Right now the point is delivery. We now own the point of sale flow and now "
    "support it. At this point the team is stable and we now ship weekly."
)

_PEOPLE = [
    # (id, name, title, years, address, skills, extra body text)
    (1, "K Naveen", "Software Engineer", 6.0, "12 South Street, T Nagar, Chennai 600017",
     ["Java", "Spring Boot", "SQL"], "Built Java microservices with Spring Boot in production."),
    (2, "Naveen Kumar S", "Data Analyst", 4.0, "45 Lake View, Velachery, Chennai 600042",
     ["Python", "SQL"], "Python reporting pipelines and SQL warehousing."),
    (3, "Najam Uddin", "DevOps Engineer", 5.0, "Banjara Hills, Hyderabad 500034",
     ["Azure", "Docker"], "Ran Azure pipelines and Azure DevOps release automation daily."),
    (4, "Gangaraju R", "Senior Software Engineer", 13.0, "8th Main Road, Manjunath Nagar, Bangalore 560010",
     ["Java", "Ansible"], "Java J2EE platform work and Ansible based DevOps activities."),
    (5, "Priya Krishnan", "Frontend Engineer", 3.0, "Adyar, Chennai 600020",
     ["React", "JavaScript"], "React and JavaScript interfaces for a dental platform."),
]


def _build_corpus() -> ModelIndex:
    engine = _HashingEngine()
    chunks: List[Dict[str, Any]] = []
    resumes: Dict[str, Dict[str, Any]] = {}

    for rid, name, title, years, address, skills, body in _PEOPLE:
        filename = f"{rid}_{name.replace(' ', '_')}.pdf"
        # A resume whose name extraction failed carries the synthetic placeholder the
        # real pipeline generates. This is the one that hijacked skill searches.
        stored_name = name if rid != 5 else f"Candidate #{rid}"
        resumes[str(rid)] = {
            "resume_id": rid, "filename": filename, "name": stored_name, "title": title,
            "total_years": years, "seniority_level": "Senior" if years >= 5 else "Mid",
            "email": f"user{rid}@example.com", "phone": None,
            "location": address.split(",", 1)[-1].strip(),
            "skills": list(skills) + ["Now", "Point"],   # the junk the harvester produced
        }
        for ordinal, (section, text) in enumerate([
            ("Summary", f"{name}\n{address}\n{_PROSE}"),
            ("Skills", "Technical Skills: " + ", ".join(skills)),
            ("Experience", f"{body} {_PROSE}"),
        ]):
            chunks.append({
                "chunk_id": len(chunks), "resume_id": rid, "ordinal": ordinal,
                "section": section, "page": 1, "filename": filename, "text": text,
            })

    vectors = engine.embed_documents([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(_DIM)
    index.add(vectors)

    # The real store reads profiles from the shared `documents.db`; a unit test supplies
    # them directly so it never depends on what happens to be on the operator's disk.
    model_index = ModelIndex("test", engine, index, chunks)
    model_index._resumes = {int(rid): profile for rid, profile in resumes.items()}
    return model_index


@pytest.fixture()
def pool(monkeypatch):
    """
    The pool every test in this module talks to, with a clean conversation.

    The chatbot reads the SHARED store — the same index AI Analysis, Ranking and Model
    Lab read — so that is the single thing a test needs to substitute.
    """
    corpus = _build_corpus()
    monkeypatch.setattr(embedding_store, "load_index", lambda *a, **k: corpus)
    return corpus


@pytest.fixture()
def lexicon(pool):
    """The pool's corpus lexicon, for tests that parse a question without asking it."""
    return pool.lexicon


@pytest.fixture()
def ask(pool):
    """Ask a question in one continuous conversation, deterministically."""
    session = f"regression-{id(pool)}"
    chat_service.reset_session(session)

    def _ask(message: str, limit: int = 5) -> Dict[str, Any]:
        return chat_service.answer(
            db=None, message=message, session_id=session, limit=limit, use_llm=False
        )
    return _ask


# --- REGRESSION: ordinary English must never become a search requirement -----
def test_now_is_not_a_skill(ask):
    """'who has now experience' searched for the skill "Now" and matched everyone."""
    result = ask("who has now experience")
    assert "Now" not in (result["intent"] or {}).get("skills", [])
    assert result["answer_type"] != "candidates" or not result["candidates"], (
        "an unrecognised word must not silently become a filter that matches the pool"
    )


def test_point_is_not_a_skill_in_a_conversational_question(ask):
    ask("who knows azure")
    result = ask("support your point why you say najam uddin is best in azure")
    assert "Point" not in (result["intent"] or {}).get("skills", [])


# --- REGRESSION: a placeholder name is not a name ---------------------------
def test_the_word_candidate_does_not_match_placeholder_named_resumes(ask):
    """
    'best candidate who is in t nagar and knows java' was read as a lookup of the
    people literally called "Candidate #124", which threw the Java requirement away.
    """
    result = ask("best candidate who is in t nagar and knows java")
    intent = result["intent"]
    assert intent["named_candidates"] == [], "the word 'candidate' is not somebody's name"
    assert [s.lower() for s in intent["skills"]] == ["java"], "the skill must survive"
    assert intent["places"], "the location must be recognised as a constraint"


def test_a_conjunction_never_answers_with_an_unrelated_person(ask):
    """The T. Nagar + Java question came back with Manjunath Nagar + Ansible."""
    result = ask("best candidate who is in t nagar and knows java")
    for candidate in result["candidates"]:
        assert candidate["location_match"], (
            f"{candidate['name']} does not satisfy the location that was asked for"
        )


def test_an_impossible_conjunction_asks_instead_of_substituting(ask):
    """
    Nobody in Velachery knows Java. The useful answer names which half failed and
    offers the measured alternatives — not a stranger who satisfies neither.
    """
    result = ask("who is in velachery and knows java")
    assert result["answer_type"] in ("clarification", "empty")
    assert not result["candidates"]
    assert "velachery" in result["answer"].lower()
    if result["clarification"]:
        counts = [o["label"] for o in result["clarification"]["options"]]
        assert counts, "an offer to relax a constraint must come with real options"


# --- REGRESSION: names are matched by token set, not by position ------------
def test_reversed_name_order_still_finds_the_person(ask):
    """The pool stores 'K Naveen'; the recruiter typed 'naveen k'."""
    result = ask("where is naveen k from")
    assert result["answer_type"] == "fact"
    assert result["fact"]["name"] == "K Naveen"


def test_an_ambiguous_first_name_asks_which_person(ask):
    """Two Naveens exist. Picking one silently is a guess the recruiter cannot see."""
    result = ask("tell me about naveen")
    assert result["answer_type"] == "clarification"
    assert result["needs_clarification"]
    labels = " ".join(o["label"] for o in result["clarification"]["options"])
    assert "K Naveen" in labels and "Naveen Kumar S" in labels
    assert result["clarification"]["allow_all"], (
        "the recruiter must be able to say 'I don't know which' and read the summaries"
    )


def test_choosing_from_the_disambiguation_answers_the_original_question(ask):
    ask("tell me about naveen")
    result = ask("the second one")
    assert result["answer_type"] == "candidates"
    assert result["candidates"][0]["name"] == "Naveen Kumar S"


def test_not_sure_which_one_summarises_all_of_them(ask):
    ask("tell me about naveen")
    result = ask("no idea, show me all of them")
    names = {c["name"] for c in result["candidates"]}
    assert {"K Naveen", "Naveen Kumar S"} <= names


def test_an_unknown_name_is_never_answered_with_a_stranger(ask):
    result = ask("tell me about Ramaswamy Venkatachalam")
    assert not result["candidates"]
    assert result["needs_clarification"] or result["answer_type"] == "clarification"


# --- REGRESSION: the conversation is part of the question -------------------
def test_a_pronoun_follow_up_is_answered_not_refused(ask):
    """'where is he from' was blocked by the scope guardrail."""
    ask("tell me about najam uddin")
    result = ask("where is he from")
    assert not result["refused"], "a follow-up about the established subject is in scope"
    assert result["answer_type"] == "fact"
    assert "hyderabad" in (result["answer"] + str(result["fact"]["value"])).lower()


def test_an_arbitrary_document_question_is_answered_from_the_resume(ask):
    """Any fact the document holds must be reachable, not just the modelled fields."""
    ask("tell me about najam uddin")
    result = ask("what has he done with azure pipelines")
    assert result["answer_type"] == "fact"
    assert result["fact"]["evidence"], "a fact answer must quote the resume"


def test_a_detail_the_resume_lacks_is_admitted_not_invented(ask):
    ask("tell me about najam uddin")
    result = ask("what is his notice period")
    assert result["answer_type"] == "fact"
    assert not result["fact"]["found"] or result["fact"]["evidence"]
    assert not result["candidates"], "a missing detail must not become a pool search"


def test_initials_are_derivable(ask):
    ask("tell me about najam uddin")
    result = ask("what are his initials")
    assert result["fact"]["value"] == "NU"


# --- REGRESSION: defending an answer must not silently re-score it ----------
def test_justifying_a_recommendation_does_not_change_its_score(ask):
    """
    'support your point why you say X is best in azure' re-ran the search, read
    "Point" as a requirement, and reported 56% for the candidate just scored at 88%.
    """
    first = ask("who is strongest in azure")
    assert first["candidates"], "precondition: the search must find somebody"
    original = first["candidates"][0]["match_percentage"]
    name = first["candidates"][0]["name"]

    second = ask(f"support your point why you say {name} is best in azure")
    assert second["answer_type"] == "explanation"
    assert second["diagnostics"].get("rescored") is False
    assert second["candidates"][0]["match_percentage"] == original, (
        "a defence of a number must not produce a different number"
    )


def test_an_explanation_corrects_an_overstated_premise(ask):
    first = ask("who is strongest in azure")
    name = first["candidates"][0]["name"]
    second = ask(f"why do you say {name} is the best at azure")
    assert "closest match" in second["answer"].lower(), (
        "the assistant claimed 'closest match on the evidence', not 'the best' — "
        "it must say so rather than defend a claim it never made"
    )


# --- Suggestions must follow from the answer that was actually given --------
def test_suggestions_are_about_the_current_answer(ask):
    ask("tell me about najam uddin")
    result = ask("where is he from")
    labels = " ".join(s["label"] for s in result["suggestions"]).lower()
    assert "najam uddin" in labels, (
        "follow-ups under a fact about one person must be about that person"
    )


def test_an_unresolvable_question_offers_the_obvious_reading(ask):
    """
    "who has now experience" has no resolvable facet — every content word is ordinary
    English. A shrug is technically correct and useless; offering the reading the
    recruiter almost certainly meant, as a choice they make, is not a guess.
    """
    result = ask("who has now experience")
    assert result["answer_type"] == "clarification"
    labels = " ".join(o["label"] for o in result["clarification"]["options"]).lower()
    assert "experience" in labels


def test_a_place_answer_does_not_include_the_postal_code(ask):
    """"Based in Hyderabad" is an answer; "based in Hyderabad 500034" is an address."""
    ask("tell me about najam uddin")
    result = ask("where is he from")
    assert result["fact"]["value"] == "Hyderabad"


# --- REGRESSION: a technology is never a place ------------------------------
@pytest.mark.parametrize("line", [
    "Languages: Java, Python, SQL, Spring Boot | Chennai 600017",
    "Skills: React, Node.js, MongoDB, Docker | Pune 411001",
])
def test_a_skills_list_beside_an_address_does_not_donate_places(line):
    """
    'java candidate' came back with skill depth 0% for everyone, because "Java" had
    been harvested as a LOCATION.

    Real CVs put a skills list and a city on the same contact line, and that line
    genuinely IS an address line — so shape alone cannot separate them. Two guards
    now do: the address REGION is narrowed to the segment carrying the postal code,
    and a term the corpus knows as a technology can never be a place.
    """
    from app.services.ai.chat_lexicon import build_lexicon

    profiles = [{"resume_id": 1, "name": "A B", "skills": ["Java", "React"], "location": None}]
    chunks = [{"resume_id": 1, "section": "Summary", "page": 1, "filename": "f",
               "text": f"A B\n{line}"}]
    lex = build_lexicon(chunks, profiles)

    for tech in ("java", "python", "sql", "spring boot", "react", "docker", "mongodb"):
        assert tech not in lex.places, f"{tech!r} was harvested as a place"
    # The real city on that same line must still be found.
    assert {"chennai", "pune"} & set(lex.places)


def test_a_bare_technology_is_read_as_a_skill_not_a_location(pool):
    """"java candidate" asks for a skill. Answering by residence is a different question."""
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent("java candidate", pool.lexicon)
    assert [s.lower() for s in intent.skills] == ["java"]
    assert intent.places == []


def test_a_real_location_still_works_alongside_a_skill(pool):
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent("java developer in t nagar", pool.lexicon)
    assert [s.lower() for s in intent.skills] == ["java"]
    assert [p.lower() for p in intent.place_labels] == ["t nagar"]


# --- A bare topic is the start of a question, not a question ----------------
def test_a_bare_skill_is_recognised_as_underspecified(pool):
    """
    Typing "java" names a topic. It does not say best, most senior, project-proven or
    nearby — so answering with a ranked list picks a reading the recruiter never chose.
    """
    from app.services.ai.chat_query_understanding import parse_intent

    assert parse_intent("java", pool.lexicon).underspecified


@pytest.mark.parametrize("phrasing", [
    "who knows java",
    "find me a java developer",
    "i need java",
    "java with 5 years experience",
    "java in chennai",
])
def test_a_real_request_is_never_treated_as_a_bare_topic(phrasing, pool):
    """The clarification must never get in the way of a question that was actually asked."""
    from app.services.ai.chat_query_understanding import parse_intent

    assert not parse_intent(phrasing, pool.lexicon).underspecified


def test_refinement_options_carry_real_counts_and_an_escape_hatch(pool):
    """
    Every option offered is measured off the matches themselves, so nothing is
    suggested that would come back empty — and one click always reaches the plain list.
    """
    from app.services.ai import chat_service
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent("java", pool.lexicon)
    candidates = [
        {"resume_id": 1, "name": "A", "total_years": 6.0, "demonstrated_skills": ["Java"]},
        {"resume_id": 2, "name": "B", "total_years": 12.0, "demonstrated_skills": ["Java"]},
        {"resume_id": 3, "name": "C", "total_years": 2.0, "demonstrated_skills": []},
        {"resume_id": 4, "name": "D", "total_years": 1.0, "demonstrated_skills": []},
    ]
    refinement = chat_service._refinement_clarification(candidates, intent, pool)

    assert refinement is not None
    labels = [o["label"] for o in refinement["options"]]
    assert any("5+ years" in label and "(2)" in label for label in labels), labels
    assert any("Proven in real projects" in label and "(2)" in label for label in labels), labels
    assert any("Just show me" in label for label in labels), (
        "a clarification the recruiter cannot skip is an obstacle, not a question"
    )


def test_a_topic_too_small_to_slice_is_simply_answered(ask):
    """Interrogating someone about two candidates is worse than answering them."""
    result = ask("java")
    assert result["answer_type"] == "candidates"
    assert result["candidates"]


def test_no_suggestion_offers_a_junk_token_as_a_skill(ask):
    result = ask("who knows java")
    for suggestion in result["suggestions"]:
        assert "now" not in suggestion["label"].lower().split()
        assert "point" not in suggestion["label"].lower().split()


# --- REGRESSION: a comparison names TWO people -------------------------------
def test_comparing_two_people_keeps_both(ask):
    """
    'compare X and Y' returned ONE candidate at 0%.

    Name resolution scored people globally and kept the single best match, so the
    person whose name contributed fewer tokens was silently discarded and the
    "comparison" ran against one candidate.
    """
    result = ask("compare najam uddin and k naveen")
    assert result["answer_type"] == "comparison"
    assert {c["name"] for c in result["candidates"]} == {"Najam Uddin", "K Naveen"}


def test_a_comparison_says_what_separates_them(ask):
    """Two scores side by side is not a comparison — the difference is the answer."""
    result = ask("compare najam uddin and k naveen")
    top = result["candidates"][0]
    assert top["only_they_have"] or top["proven_where_others_only_claim"], (
        "a comparison must identify what one candidate has that the other does not"
    )
    assert any(word in result["answer"].lower() for word in ("brings", "evidences", "against"))


def test_one_ambiguous_name_does_not_discard_the_other_person(pool):
    """
    'why is anita better than hemankshree' asked "3 people answer to that name?" —
    pooling everyone who tied on one token hit into a single ambiguity, and losing
    the fact that TWO different people had been named.
    """
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent("why is najam uddin better than naveen", pool.lexicon)
    assert [p.name for p in intent.people] == ["Najam Uddin"], "the clear name stays resolved"
    assert {p.name for p in intent.ambiguous_options} == {"K Naveen", "Naveen Kumar S"}


# --- REGRESSION: cards must not all read the same ---------------------------
def test_each_candidate_gets_its_own_points(ask):
    """Every card carried the same templated sentence, which told a recruiter nothing."""
    result = ask("who knows java")
    assert len(result["candidates"]) >= 2
    for c in result["candidates"]:
        assert c["verdict"], "each card needs a headline judgement"
        assert c["highlights"], "each card needs scannable points, not a paragraph"


# --- REGRESSION: the conversation remembers who, not just the last turn ------
def test_a_name_discussed_earlier_is_still_known_later(ask):
    """
    Asking about somebody again after several unrelated turns must not need the full
    name retyped. One "current subject" is not memory: it forgets the moment the
    recruiter looks at anybody else.
    """
    ask("tell me about k naveen")
    for filler in ("who knows azure", "who is in hyderabad", "who knows docker",
                   "most experienced candidates", "who knows java"):
        ask(filler)
    result = ask("how about naveen")
    assert result["answer_type"] == "candidates"
    assert result["candidates"][0]["name"] == "K Naveen"


def test_memory_never_guesses_through_a_real_ambiguity(ask):
    """Two discussed people answering to the same word must still produce a question."""
    ask("tell me about k naveen")
    ask("tell me about naveen kumar s")
    result = ask("how about naveen")
    assert result["answer_type"] == "clarification"


# --- REGRESSION: an experience bar has a direction --------------------------
# "best data science candidate in chennai with less than 5 years experience" returned
# an 11-year Cloud Architect in Gurugram and scored his experience fit at 100%. The
# years pattern matched the bare "5 years" inside "less than 5 years" and stored it as
# a MINIMUM, so the more experience a candidate had, the better they scored against a
# request for less.
@pytest.mark.parametrize("phrasing", [
    "less than 5 years experience",
    "under 5 years experience",
    "below 5 years experience",
    "fewer than 5 years experience",
    "at most 5 years experience",
    "no more than 5 years experience",
    "5 years or less experience",
    "5 yrs max experience",
])
def test_an_upper_experience_bound_is_never_read_as_a_minimum(lexicon, phrasing):
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent(f"java developer with {phrasing}", lexicon)
    assert intent.max_years == 5.0, phrasing
    assert intent.min_years is None, phrasing


@pytest.mark.parametrize("phrasing,expected", [
    ("5+ years experience", 5.0),
    ("at least 5 years experience", 5.0),
    ("more than 5 years experience", 5.0),
    ("minimum 5 years experience", 5.0),
])
def test_a_lower_experience_bound_still_reads_as_a_minimum(lexicon, phrasing, expected):
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent(f"java developer with {phrasing}", lexicon)
    assert intent.min_years == expected
    assert intent.max_years is None


def test_too_much_experience_is_a_mismatch_when_a_ceiling_was_asked_for(lexicon):
    """The scoring half of the same bug: 11 years against a 5-year ceiling must not
    score as a perfect experience fit."""
    from app.services.ai.chat_query_understanding import QueryIntent
    from app.services.ai.chat_retrieval_service import _experience_fit

    fit, note = _experience_fit(11.0, None, 5.0)
    assert fit < 0.5
    assert "over the limit" in note

    inside, note_inside = _experience_fit(3.0, None, 5.0)
    assert inside == 1.0
    assert "within" in note_inside

    assert QueryIntent(raw="x", max_years=5.0).has_facets


# --- REGRESSION: a city is not a skill --------------------------------------
def test_a_city_is_a_place_not_a_searchable_skill():
    """
    "who is strongest in .net and location bangalore" was understood as
    skills=[NET, Bangalore] and filtered nobody, so a candidate from Andhra Pradesh
    came back with "Bangalore — proven in the education, experience and summary
    sections (71% depth)".

    Resumes shout their cities in capitals, and the skill harvester admits ALL-CAPS
    tokens, so BANGALORE entered the skill vocabulary — and the technology-is-never-a-
    place guard then blocked it from the gazetteer, because it now looked like a skill.
    Address evidence has to be able to outweigh that.
    """
    from app.services.ai.chat_lexicon import build_lexicon

    profiles = [
        {"resume_id": 1, "name": "A B", "skills": ["BANGALORE", "C#"],
         "location": "Bangalore, India"},
        {"resume_id": 2, "name": "C D", "skills": ["Java"], "location": None},
    ]
    chunks = [
        {"resume_id": 1, "section": "Summary", "page": 1, "filename": "a",
         "text": "A B\nAddress: Bangalore, India\nSkills: BANGALORE, C#"},
        {"resume_id": 2, "section": "Summary", "page": 1, "filename": "b",
         "text": "C D\nJava developer"},
    ]
    lex = build_lexicon(chunks, profiles)

    assert "bangalore" in lex.places
    assert "bangalore" not in lex.skill_terms, "a city must not stay searchable as a skill"
    assert lex.find_places("strongest in .net and location bangalore") == ["bangalore"]


def test_a_library_in_a_broken_location_field_does_not_become_a_city():
    """
    The upstream location extractor writes a skills line into `location` when a CV has
    no address — "JUnit, Mockito" is a real value in this pool. A declared location
    therefore still has to outweigh how often the corpus calls the same word a skill.
    """
    from app.services.ai.chat_lexicon import build_lexicon

    profiles = [
        {"resume_id": 1, "name": "A B", "skills": ["JUnit", "Mockito"],
         "location": "JUnit, Mockito"},
        {"resume_id": 2, "name": "C D", "skills": ["Mockito"], "location": "JUnit, Mockito"},
    ]
    chunks = [{"resume_id": i, "section": "Skills", "page": 1, "filename": "f",
               "text": "Testing: JUnit, Mockito"} for i in (1, 2)]
    lex = build_lexicon(chunks, profiles)

    assert "mockito" not in lex.places
    assert "junit" not in lex.places


# --- REGRESSION: a shouted word is not an acronym ---------------------------
def test_a_capitalised_heading_or_city_is_not_harvested_as_a_skill():
    """
    "react developer in chennai" was understood as skills=[react, DEVELOPER, CHENNAI]
    and filtered nobody, so every candidate scored ~88% "skill depth" on two words
    that are not skills — and a Power BI analyst with no React at all ranked.

    Resumes capitalise headings, job titles and cities, and the harvester read any run
    of capitals as an acronym. An acronym is short; a shouted word is long.
    """
    from app.services.ai.chat_lexicon import build_lexicon

    profiles = [{
        "resume_id": 1, "name": "A B",
        "skills": ["REACT", "DEVELOPER", "CHENNAI", "AWS", "SQL", "HTML"],
        "location": "Chennai, India",
    }]
    chunks = [{"resume_id": 1, "section": "Skills", "page": 1, "filename": "f",
               "text": "A B\nAddress: Chennai, India\nDEVELOPER\nSkills: AWS, SQL, HTML"}]
    lex = build_lexicon(chunks, profiles)

    # Short runs of capitals are real acronyms and survive.
    for acronym in ("aws", "sql", "html"):
        assert acronym in lex.skill_terms, acronym
    # Long ones are shouted words, not technologies.
    assert "developer" not in lex.skill_terms
    assert "chennai" not in lex.skill_terms
    assert "chennai" in lex.places


def test_a_role_and_a_city_do_not_become_search_requirements(lexicon):
    """The whole query, end to end: one skill and one place, nothing else."""
    from app.services.ai.chat_query_understanding import parse_intent

    intent = parse_intent("react developer in chennai", lexicon)
    assert [s.lower() for s in intent.skills] == ["react"]
    assert "developer" not in " ".join(intent.skills).lower()


# --- REGRESSION: one answer must cost one LLM round-trip, not three ---------
def test_the_three_llm_stages_run_concurrently(pool, monkeypatch):
    """
    A single question took ~29 seconds against an 8B model because the summary, the
    per-candidate reports and the follow-up suggestions were three sequential LLM
    calls. None of them reads another's output, so the wait should be the slowest
    call, not their sum.

    Each stage is stubbed with a deliberate delay: run in series this takes 0.9s, in
    parallel a little over 0.3s. The assertion sits between the two, so a regression
    to sequential execution fails rather than merely getting slower.
    """
    import time as _time

    delay = 0.3

    def _slow(value):
        def _fn(*args, **kwargs):
            _time.sleep(delay)
            return value
        return _fn

    monkeypatch.setattr(chat_service, "_llm_summary", _slow("A summary."))
    monkeypatch.setattr(chat_service, "_llm_candidate_reports", _slow({}))
    monkeypatch.setattr(chat_service, "_llm_suggestions", _slow([]))

    session = "concurrency-check"
    chat_service.reset_session(session)
    started = _time.monotonic()
    result = chat_service.answer(
        db=None, message="who knows java", session_id=session, limit=5, use_llm=True
    )
    elapsed = _time.monotonic() - started

    assert result["answer_type"] == "candidates"
    assert elapsed < delay * 2, (
        f"the three LLM stages took {elapsed:.2f}s — that is sequential, not concurrent"
    )


def test_one_failing_llm_stage_does_not_sink_the_answer(pool, monkeypatch):
    """A stage that raises must fall back to the deterministic text, not 500."""
    def _boom(*args, **kwargs):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(chat_service, "_llm_summary", _boom)
    monkeypatch.setattr(chat_service, "_llm_candidate_reports", _boom)
    monkeypatch.setattr(chat_service, "_llm_suggestions", _boom)

    session = "llm-failure"
    chat_service.reset_session(session)
    result = chat_service.answer(
        db=None, message="who knows java", session_id=session, limit=5, use_llm=True
    )

    assert result["answer"]
    assert result["diagnostics"]["reasoning_engine"] == "deterministic"


# --- The chatbot scores with the Match Score parameters, not its own algorithm ---
def test_chat_scores_with_renormalised_match_score_weights(pool):
    """
    The platform must not carry two scoring algorithms. Chat used to run
    `coverage×0.5 + evidence×0.3 + experience×0.2`, so the same candidate got one
    number on the Analysis screen and a different one in chat.

    A question is not a Job Description, so only the parameters the recruiter stated
    are scored and their Match Score weights are renormalised across them. Skills-only
    means Skill 20 + Technology 14, rescaled — a perfect match reads 100, not 34.
    """
    from app.services.ai.chat_query_understanding import QueryIntent
    from app.services.ai.chat_retrieval_service import score_request

    intent = QueryIntent(raw="java", skills=["java"])
    score, params = score_request(
        requested=["java"], matched=["java"], demonstrated=["java"],
        place_ok=True, experience_fit=1.0, intent=intent, fallback=0.0,
    )
    assert score == pytest.approx(1.0)
    assert {p["key"] for p in params} == {"skill", "technology"}
    assert sum(p["weight"] for p in params) == pytest.approx(1.0, abs=1e-3)
    # 20 and 14 renormalised over 34.
    weights = {p["key"]: p["weight"] for p in params}
    assert weights["skill"] == pytest.approx(20 / 34, abs=1e-3)
    assert weights["technology"] == pytest.approx(14 / 34, abs=1e-3)


def test_asking_for_many_skills_and_finding_few_scores_proportionally(pool):
    """"If we asked for many skills and only few found we mark based on that"."""
    from app.services.ai.chat_query_understanding import QueryIntent
    from app.services.ai.chat_retrieval_service import score_request

    intent = QueryIntent(raw="x", skills=["java", "python", "docker", "aws"])
    score, params = score_request(
        requested=["java", "python", "docker", "aws"],
        matched=["java", "python"], demonstrated=["java"],
        place_ok=True, experience_fit=1.0, intent=intent, fallback=0.0,
    )
    by_key = {p["key"]: p for p in params}
    assert by_key["skill"]["score"] == 50.0        # 2 of 4 evidenced
    assert by_key["technology"]["score"] == 25.0   # 1 of 4 proven in real work
    assert 0.0 < score < 0.5


def test_location_is_marked_all_or_nothing(pool):
    """"If location matches it gets 100%, if not it gets 0" — there is no half a city."""
    from app.services.ai.chat_query_understanding import QueryIntent
    from app.services.ai.chat_retrieval_service import score_request

    intent = QueryIntent(raw="x", skills=["react"], places=["chennai"],
                         place_labels=["Chennai"])
    kwargs = dict(requested=["react"], matched=["react"], demonstrated=["react"],
                  experience_fit=1.0, intent=intent, fallback=0.0)

    here, params_here = score_request(place_ok=True, **kwargs)
    away, params_away = score_request(place_ok=False, **kwargs)

    assert {p["key"]: p["score"] for p in params_here}["location"] == 100.0
    assert {p["key"]: p["score"] for p in params_away}["location"] == 0.0
    assert here == pytest.approx(1.0)
    # Skill 20 + Technology 14 survive; Location 8 of 42 is lost.
    assert away == pytest.approx(34 / 42, abs=1e-3)


def test_an_overview_question_gets_no_invented_percentage(pool):
    """Nothing was asked for, so there is no requirement to score against."""
    from app.services.ai.chat_query_understanding import QueryIntent
    from app.services.ai.chat_retrieval_service import score_request

    score, params = score_request(
        requested=[], matched=[], demonstrated=[], place_ok=True,
        experience_fit=1.0, intent=QueryIntent(raw="tell me about x"), fallback=0.42,
    )
    assert params == []
    assert score == pytest.approx(0.42)


def test_a_misspelt_skill_earns_partial_credit_not_zero(pool):
    """
    Shabeer is a real React developer whose resume writes "Reat js". Retrieval cannot
    find that — BM25 needs the literal token and one dropped letter is not reliably
    close in embedding space — so chat scored him as having no React at all, while the
    JD-side Match Score, reading the same specification, gave him partial credit.
    """
    from app.services.ai.chat_retrieval_service import _misspelled_matches, score_request
    from app.services.ai.chat_query_understanding import QueryIntent

    profile = {"skills": ["Reat js", "HTML 5", "Bootstrap", "SQL Server"]}
    assert _misspelled_matches(["react"], [], profile) == ["react"]
    # A genuinely absent skill is still absent.
    assert _misspelled_matches(["kubernetes"], [], profile) == []
    # A skill already matched is not double-counted.
    assert _misspelled_matches(["react"], ["react"], profile) == []

    intent = QueryIntent(raw="react", skills=["react"])
    scored, params = score_request(
        requested=["react"], matched=[], demonstrated=[], place_ok=True,
        experience_fit=1.0, intent=intent, fallback=0.0, misspelled=["react"],
    )
    skill = next(p for p in params if p["key"] == "skill")
    assert skill["score"] == pytest.approx(60.0)     # partial, not 0 and not 100
    assert "spelling variant" in skill["basis"]


def test_a_short_lookalike_is_not_treated_as_a_typo(pool):
    """"Go" and "god" are close in characters and unrelated in meaning."""
    from app.services.ai.chat_retrieval_service import _misspelled_matches

    assert _misspelled_matches(["go"], [], {"skills": ["god"]}) == []


# --- REGRESSION: two-letter requirements, real marks, and "show me 10" ------
def test_two_letter_acronyms_are_real_requirements():
    """
    "communication skill for hr role" searched for Communication ALONE — a blanket
    three-character floor dropped "HR" — so five people with no HR background came
    back at 100%. A whole class of real requirements is two letters: HR, QA, ML, AI.
    """
    from app.services.ai.chat_lexicon import build_lexicon

    profiles = [{"resume_id": 1, "name": "A B",
                 "skills": ["HR", "QA", "ML", "Communication"], "location": None}]
    chunks = [{"resume_id": 1, "section": "Skills", "page": 1, "filename": "f",
               "text": "HR Skills: Talent Acquisition. QA and ML exposure."}]
    lex = build_lexicon(chunks, profiles)

    for acronym in ("hr", "qa", "ml"):
        assert acronym in lex.skill_terms, acronym
    # Two-letter ordinary English stays out.
    for word in ("of", "in", "to"):
        assert word not in lex.skill_terms, word


def test_semantic_nearness_alone_does_not_evidence_a_skill():
    """
    Dense retrieval answers "what is this passage about", and for a soft requirement
    like Communication almost every resume is close enough. A Mulesoft architect was
    returned at 100% on passages that never mention communication or HR.

    Retrieval stays semantic; confirmation is lexical.
    """
    from app.services.ai.chat_retrieval_service import _confirms

    assert _confirms("Excellent written and verbal communication skills.", "Communication")
    assert _confirms("HR Skills: Talent Acquisition & Onboarding", "HR")
    # About teamwork, but never says the word.
    assert not _confirms("Collaborated with stakeholders across delivery teams.", "Communication")
    assert not _confirms("Designed Mulesoft integration flows and APIs.", "HR")
    # A multi-word skill confirms on any content word.
    assert _confirms("Built services with Spring and Hibernate.", "Spring Boot")
    # And the pool's own misspellings still confirm.
    assert _confirms("Developed the portal using Reat js and .net core", "React")


@pytest.mark.parametrize("question,expected", [
    ("show me 10 candidates with java", 10),
    ("top 3 react developers", 3),
    ("give me 15 profiles", 15),
    ("list 7 resumes", 7),
])
def test_a_requested_count_is_honoured(lexicon, question, expected):
    """"Display 10 candidates" returned 5 — the number was in the sentence, ignored."""
    from app.services.ai.chat_query_understanding import parse_intent

    assert parse_intent(question, lexicon).wanted_count == expected


@pytest.mark.parametrize("question", [
    "java developers with 10 years experience",
    "candidates who know react 18",
    "less than 5 years experience",
])
def test_a_number_that_is_not_a_count_is_not_read_as_one(lexicon, question):
    """"10 years" and "React 18" are not requests for 10 or 18 candidates."""
    from app.services.ai.chat_query_understanding import parse_intent

    assert parse_intent(question, lexicon).wanted_count is None
