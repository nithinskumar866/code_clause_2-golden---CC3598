"""
Regressions found by scoring a closely matching resume and reading the report.

The candidate had PostgreSQL in their Experience section and was told it was missing,
was flagged as over-claiming Docker they had used, and was marked down for "skills"
called Minimum, Joining and Mentor. The first two share one cause: a one-word query
against a long passage scores below the similarity floor even when the word is in it.
"""
import pytest

from app.services.ai import authenticity_service, job_profile_extractor, retrieval_service, skill_semantics_service
from app.services.ai import text_matching
from app.services.ai.jd_requirement_extractor import extract_requirements

JD = """Senior Backend Engineer
Company: Northwind Labs
Location: Bengaluru, India
Responsibilities
Design and build Python microservices
Own PostgreSQL schemas and query performance
Mentor junior engineers
Required Skills
Python
Snowflake
Qualifications
Minimum 5 years of backend experience. Bachelor's degree in Computer Science or a related field. Joining within 30 days."""


@pytest.mark.parametrize("text, term, expected", [
    ("Designed PostgreSQL schemas", "postgresql", True),
    ("Built apps in ReactJS", "react", True),
    ("Built APIs with node.js", "node", True),
    ("Wrote C++ services", "c++", True),
    ("Led the AI platform", "ai", True),
    ("Wrote services in Go", "go", True),
    ("Worked in JavaScript", "java", False),
    ("Moved to NoSQL stores", "sql", False),
    ("Ready to go live", "go", False),
    ("Said it was fair and said no more", "ai", False),
    ("Wrote C++ services", "c", False),
    ("", "python", False),
])
def test_mentions_term(text, term, expected):
    assert text_matching.mentions_term(text, term) is expected


@pytest.mark.parametrize("required, held, low, high", [
    ("Bachelor's degree in Computer Science or a related field", "B.Tech in Computer Science, 2016", 1.0, 1.0),
    ("Bachelor's degree in Computer Science", "MS in Computer Science", 1.0, 1.0),
    ("B.E in Computer Science", "B.A History", 0.4, 0.6),
    ("Master's degree in Computer Science", "B.Tech in Computer Science", 0.4, 0.6),
    ("PhD in Physics", "MBA in Marketing", 0.0, 0.1),
])
def test_qualification_similarity(required, held, low, high):
    assert low <= text_matching.qualification_similarity(required, held) <= high


class _Engine:
    min_similarity = 0.62


class _Index:
    """The slice of ModelIndex retrieval reads, with cosines chosen by the test."""

    model = "fake"
    engine = _Engine()

    def __init__(self, chunks, cosine_by_row):
        self.chunks = chunks
        self.cosine_by_row = cosine_by_row
        self.rows_by_resume = {1: list(range(len(chunks)))}

    def search(self, query, top_k, resume_ids=None):
        return sorted(self.cosine_by_row.items(), key=lambda p: p[1], reverse=True)[:top_k]


def _chunk(row, section, text):
    return {"text": text, "section": section, "page": 1, "filename": "r.docx", "ordinal": row, "chunk_id": row}


def _index():
    return _Index(
        [
            _chunk(0, "Experience", "Designed PostgreSQL schemas and deployed with Docker on AWS."),
            _chunk(1, "Skills", "Python, FastAPI, PostgreSQL, Docker, AWS"),
        ],
        {0: 0.50, 1: 0.58},
    )


def test_a_chunk_naming_the_requirement_counts_below_the_floor():
    result = retrieval_service.retrieve_evidence_from_store(_index(), 1, ["postgresql"])[0]
    assert {m["section"] for m in result["matches"]} == {"Experience", "Skills"}


def test_a_chunk_not_naming_the_requirement_is_still_held_to_the_floor():
    result = retrieval_service.retrieve_evidence_from_store(_index(), 1, ["salesforce"])[0]
    assert result["matches"] == []


def test_a_skills_list_that_only_resembles_a_requirement_is_not_evidence():
    index = _Index([_chunk(0, "Skills", "Python, Flask, Docker")], {0: 0.72})
    assert retrieval_service.retrieve_evidence_from_store(index, 1, ["kubernetes"])[0]["matches"] == []


def test_prose_that_clears_the_floor_is_evidence_without_naming_the_requirement():
    index = _Index([_chunk(0, "Experience", "Ran sprint ceremonies and stakeholder reviews.")], {0: 0.70})
    assert retrieval_service.retrieve_evidence_from_store(index, 1, ["communication"])[0]["matches"]


def test_a_skill_used_in_experience_is_not_flagged_as_over_claimed():
    evidence = retrieval_service.retrieve_evidence_from_store(_index(), 1, ["docker"])
    assessment = authenticity_service.assess_authenticity({"retrieval_results": evidence}).assessment
    assert assessment.over_claimed_skills == []


def test_a_skill_only_in_the_list_is_still_flagged_as_over_claimed():
    index = _Index([_chunk(0, "Skills", "Python, Docker")], {0: 0.70})
    evidence = retrieval_service.retrieve_evidence_from_store(index, 1, ["docker"])
    assessment = authenticity_service.assess_authenticity({"retrieval_results": evidence}).assessment
    assert assessment.over_claimed_skills == ["docker"]


@pytest.mark.parametrize("word", ["Minimum", "Mentor", "Joining", "Bachelor", "Computer Science", "Own"])
def test_ordinary_sentence_words_are_not_requirements(word):
    assert word.lower() not in {r.lower() for r in extract_requirements(JD)}


@pytest.mark.parametrize("jd, expected", [
    (JD, "Snowflake"),
    ("Required Skills\nSnowflake, Airflow", "Airflow"),
    ("Required Skills\nSnowflake, Airflow", "Snowflake"),
    ("Required Skills\nSnowflake and Airflow", "Snowflake"),
    ("Required Skills\nCertified Scrum Master", "Certified Scrum Master"),
    ("Requirements\nSnowflake experience required", "Snowflake"),
])
def test_real_names_survive_in_the_opening_position(jd, expected):
    assert expected.lower() in {r.lower() for r in extract_requirements(jd)}


def test_a_degree_in_a_long_paragraph_is_still_the_education_bar():
    profile = job_profile_extractor.extract_job_profile(JD)
    assert profile.education and "Bachelor" in profile.education


@pytest.mark.parametrize("requirement, skill", [("mysql", "sql"), ("sql", "postgresql")])
def test_a_skill_family_is_equivalent(requirement, skill):
    _, _, relation = skill_semantics_service.find_related_skill(requirement, [skill])
    assert relation == "equivalent"


@pytest.mark.parametrize("requirement, skill", [
    ("nosql", "sql"), ("javascript", "java"), ("xml", "html"), ("django", "python"), ("preact", "react"),
])
def test_names_that_only_look_or_score_alike_are_not_equivalent(requirement, skill):
    _, _, relation = skill_semantics_service.find_related_skill(requirement, [skill])
    assert relation != "equivalent"
