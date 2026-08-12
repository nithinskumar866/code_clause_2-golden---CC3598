"""
Match Score regression suite.

Covers the three things that must hold for the score to be trustworthy:

  1. every parameter behaves as specified, including its neutral case
  2. the weighted formula is exactly the specification, to one decimal
  3. the extractors feed it the right facts from real-shaped documents

`semantic=False` throughout. The embedding-backed equivalence check is a *rescue*
layer on top of string matching; letting it run here would make these assertions
depend on a 1.3 GB model's opinion, and a scoring test that cannot be reproduced
without a GPU is not a scoring test.
"""
from datetime import datetime

import pytest

from app.core.config import settings
from app.services.ai import text_matching as tm
from app.services.ai.candidate_facets_service import (
    CandidateFacets,
    declared_skills,
    extract_candidate_facets,
)
from app.services.ai.job_profile_extractor import JobProfile, extract_job_profile
from app.services.ai.match_scoring_service import band_for, compute_match_score

NOW = datetime(2026, 8, 6)


def _score(job: JobProfile, facets: CandidateFacets):
    return compute_match_score(job, facets, as_of=NOW, semantic=False)


def _param(result, key):
    return next(p for p in result.parameters if p.key == key)


def _only(key, job_kwargs=None, facet_kwargs=None):
    """Isolate one parameter: score a job/candidate pair and return just that row."""
    job = JobProfile(**(job_kwargs or {}))
    facets = CandidateFacets(**(facet_kwargs or {}))
    return _param(_score(job, facets), key)


# --------------------------------------------------------------------------
# The formula itself
# --------------------------------------------------------------------------

def test_weights_are_the_agreed_specification():
    assert settings.MATCH_WEIGHT_SKILL == 0.20
    assert settings.MATCH_WEIGHT_EXPERIENCE == 0.15
    assert settings.MATCH_WEIGHT_TECHNOLOGY == 0.14
    assert settings.MATCH_WEIGHT_DESIGNATION == 0.14
    assert settings.MATCH_WEIGHT_INDUSTRY == 0.09
    assert settings.MATCH_WEIGHT_EDUCATION == 0.08
    assert settings.MATCH_WEIGHT_LOCATION == 0.08
    assert settings.MATCH_WEIGHT_AVAILABILITY == 0.07
    assert settings.MATCH_WEIGHT_FRESHNESS == 0.05


def test_weights_sum_to_one():
    result = _score(JobProfile(), CandidateFacets())
    assert round(sum(p.weight for p in result.parameters), 10) == 1.0
    assert len(result.parameters) == 9


def test_final_score_is_the_weighted_sum_to_one_decimal():
    result = _score(JobProfile(), CandidateFacets())
    expected = round(sum(p.score * p.weight for p in result.parameters), 1)
    assert result.score == expected
    # One decimal is the agreed reporting precision, not a rounded integer.
    assert result.score == round(result.score, 1)


def test_every_parameter_reports_its_own_contribution():
    result = _score(JobProfile(), CandidateFacets())
    for p in result.parameters:
        assert p.contribution == round(p.score * p.weight, 1)
        assert p.basis, f"{p.key} must explain itself"


def test_score_is_clamped_into_range():
    for result in (
        _score(JobProfile(), CandidateFacets()),
        _score(
            JobProfile(required_skills=["python"], min_years=20.0, title="Architect",
                       location="Delhi", industry="Healthcare", education="PhD",
                       availability_weeks=0.0),
            CandidateFacets(total_years=0.0, location="Chennai", title="Intern",
                            availability_weeks=52.0, uploaded_at=datetime(2000, 1, 1)),
        ),
    ):
        assert 0.0 <= result.score <= 100.0
        assert all(0.0 <= p.score <= 100.0 for p in result.parameters)


@pytest.mark.parametrize(
    "score,band",
    [(100.0, "Excellent fit"), (85.0, "Excellent fit"), (84.9, "Strong fit"),
     (70.0, "Strong fit"), (69.9, "Moderate fit"), (50.0, "Moderate fit"),
     (49.9, "Weak fit"), (0.0, "Weak fit")],
)
def test_interpretation_bands(score, band):
    assert band_for(score) == band


def test_the_same_inputs_always_produce_the_same_score():
    job = JobProfile(required_skills=["Python", "Docker"], min_years=3.0, max_years=6.0,
                     title="Backend Engineer", location="Chennai")
    facets = CandidateFacets(primary_skills=["python"], technologies=["docker"],
                             total_years=4.0, title="Backend Engineer", location="Chennai",
                             uploaded_at=datetime(2026, 7, 1))
    assert _score(job, facets).score == _score(job, facets).score


# --------------------------------------------------------------------------
# 1. Skill Match
# --------------------------------------------------------------------------

def test_skill_match_is_the_share_of_requirements_held():
    p = _only("skill",
              {"required_skills": ["Python", "Docker", "Redis", "Kafka"]},
              {"primary_skills": ["python", "docker", "redis"]})
    assert p.score == 75.0


def test_spelling_conventions_do_not_break_a_skill_match():
    # ReactJS / React.js / react-js are one skill under any spelling convention.
    for written in ("ReactJS", "React.js", "react-js", "REACT"):
        p = _only("skill", {"required_skills": ["React"]}, {"primary_skills": [written]})
        assert p.score == 100.0, written


def test_a_misspelt_skill_earns_partial_credit_not_full():
    p = _only("skill", {"required_skills": ["Kubernetes"]}, {"primary_skills": ["kubernets"]})
    assert p.score == pytest.approx(settings.MATCH_PARTIAL_SKILL_CREDIT * 100)
    assert "spelling variant" in p.basis


def test_a_different_skill_is_not_a_misspelling():
    # Short tokens sit above any character-similarity threshold while naming
    # completely different things; partial credit must not fire on them.
    p = _only("skill", {"required_skills": ["Go"]}, {"primary_skills": ["god"]})
    assert p.score == 0.0


def test_skill_match_names_what_is_missing():
    p = _only("skill", {"required_skills": ["Python", "Kafka"]}, {"primary_skills": ["python"]})
    assert "Kafka" in p.basis


def test_a_job_with_no_stated_skills_scores_neutrally():
    p = _only("skill", {}, {"primary_skills": ["python"]})
    assert p.neutral and p.score == settings.MATCH_NEUTRAL_SKILL


# --------------------------------------------------------------------------
# 2. Technology Match
# --------------------------------------------------------------------------

def test_technology_match_reads_only_demonstrated_work():
    # Listed in Skills but never used: full credit on Skill, none on Technology.
    job = {"required_skills": ["Kubernetes"]}
    facets = {"primary_skills": ["kubernetes"], "technologies": []}
    assert _only("skill", job, facets).score == 100.0
    assert _only("technology", job, facets).score == 0.0


def test_technology_match_credits_experience_and_projects():
    p = _only("technology",
              {"required_skills": ["Docker", "Terraform"]},
              {"technologies": ["docker"]})
    assert p.score == 50.0


# --------------------------------------------------------------------------
# 3. Designation Match
# --------------------------------------------------------------------------

def test_designation_ignores_seniority_decoration():
    p = _only("designation", {"title": "Backend Engineer"},
              {"title": "Senior Backend Engineer"})
    assert p.score == 100.0


def test_designation_is_similarity_not_string_equality():
    p = _only("designation", {"title": "Backend Engineer"}, {"title": "Backend Developer"})
    assert 0.0 < p.score < 100.0


def test_an_unrelated_title_scores_low():
    p = _only("designation", {"title": "Backend Engineer"}, {"title": "Graphic Designer"})
    assert p.score < 50.0


def test_designation_is_neutral_when_either_side_is_silent():
    assert _only("designation", {}, {"title": "Engineer"}).neutral
    assert _only("designation", {"title": "Engineer"}, {}).neutral


# --------------------------------------------------------------------------
# 4. Experience Match
# --------------------------------------------------------------------------

def test_experience_inside_the_range_is_full_marks():
    p = _only("experience", {"min_years": 3.0, "max_years": 6.0}, {"total_years": 4.0})
    assert p.score == 100.0


@pytest.mark.parametrize("years,expected", [(2.0, 85.0), (1.0, 70.0), (0.0, 55.0)])
def test_below_the_range_costs_fifteen_points_per_year(years, expected):
    p = _only("experience", {"min_years": 3.0, "max_years": 6.0}, {"total_years": years})
    assert p.score == expected


def test_above_the_range_costs_the_same_per_year():
    p = _only("experience", {"min_years": 3.0, "max_years": 6.0}, {"total_years": 8.0})
    assert p.score == 70.0


def test_an_open_ended_requirement_has_no_upper_penalty():
    p = _only("experience", {"min_years": 5.0}, {"total_years": 30.0})
    assert p.score == 100.0


def test_experience_is_neutral_when_unknown():
    assert _only("experience", {}, {"total_years": 5.0}).neutral
    assert _only("experience", {"min_years": 5.0}, {}).neutral


# --------------------------------------------------------------------------
# 5. Industry Match
# --------------------------------------------------------------------------

def test_industry_matches_against_the_work_history():
    p = _only("industry", {"industry": "FinTech"},
              {"experience_text": "Built payment rails for a fintech lender."})
    assert p.score == 100.0


def test_an_absent_industry_scores_low_but_not_zero_weighted():
    p = _only("industry", {"industry": "Healthcare"},
              {"experience_text": "Built payment rails for a fintech lender."})
    assert p.score < 50.0


def test_industry_is_neutral_when_the_jd_does_not_name_one():
    p = _only("industry", {}, {"experience_text": "anything"})
    assert p.neutral and p.score == settings.MATCH_NEUTRAL_INDUSTRY


# --------------------------------------------------------------------------
# 6. Education Match
# --------------------------------------------------------------------------

def test_education_matches_degree_and_field():
    p = _only("education", {"education": "B.E in Computer Science"},
              {"education": "B.E in Computer Science, Anna University"})
    assert p.score == 100.0


def test_a_different_field_scores_lower():
    strong = _only("education", {"education": "B.E Computer Science"},
                   {"education": "B.E Computer Science"}).score
    weak = _only("education", {"education": "B.E Computer Science"},
                 {"education": "B.A History"}).score
    assert weak < strong


def test_education_is_neutral_when_the_jd_is_silent():
    assert _only("education", {}, {"education": "B.E"}).neutral


# --------------------------------------------------------------------------
# 7. Location Match
# --------------------------------------------------------------------------

def test_a_remote_role_is_always_a_full_location_match():
    p = _only("location", {"is_remote": True}, {"location": "Reykjavik"})
    assert p.score == 100.0


def test_the_same_city_is_a_full_match():
    p = _only("location", {"location": "Chennai"}, {"location": "Chennai, Tamil Nadu"})
    assert p.score == 100.0


def test_a_different_city_scores_twenty_not_zero():
    # Relocation is possible, so a different city is a penalty, not a disqualification.
    p = _only("location", {"location": "Chennai"}, {"location": "Delhi"})
    assert p.score == settings.MATCH_LOCATION_DIFFERENT


def test_a_missing_location_is_neutral_on_both_sides():
    assert _only("location", {}, {"location": "Chennai"}).neutral
    assert _only("location", {"location": "Chennai"}, {}).neutral
    assert _only("location", {"location": "Chennai"}, {}).score == settings.MATCH_NEUTRAL_LOCATION


# --------------------------------------------------------------------------
# 8. Availability Match
# --------------------------------------------------------------------------

def test_joining_inside_the_window_is_full_marks():
    p = _only("availability", {"availability_weeks": 4.0}, {"availability_weeks": 2.0})
    assert p.score == 100.0


@pytest.mark.parametrize("weeks,expected", [(5.0, 90.0), (6.0, 80.0), (8.0, 60.0)])
def test_every_late_week_costs_ten_points(weeks, expected):
    p = _only("availability", {"availability_weeks": 4.0}, {"availability_weeks": weeks})
    assert p.score == expected


def test_a_silent_resume_scores_forty_when_the_job_asked():
    p = _only("availability", {"availability_weeks": 4.0}, {})
    assert p.score == settings.MATCH_MISSING_AVAILABILITY
    assert not p.neutral


def test_a_job_with_no_deadline_is_neutral():
    p = _only("availability", {}, {})
    assert p.neutral and p.score == settings.MATCH_NEUTRAL_AVAILABILITY


# --------------------------------------------------------------------------
# 9. Resume Freshness
# --------------------------------------------------------------------------

def test_a_resume_added_today_is_fully_fresh():
    p = _only("freshness", {}, {"uploaded_at": NOW})
    assert p.score == 100.0


def test_freshness_decays_towards_the_floor():
    recent = _only("freshness", {}, {"uploaded_at": datetime(2026, 5, 6)}).score
    older = _only("freshness", {}, {"uploaded_at": datetime(2025, 8, 6)}).score
    ancient = _only("freshness", {}, {"uploaded_at": datetime(2015, 1, 1)}).score
    assert 100.0 > recent > older
    assert ancient == settings.MATCH_FRESHNESS_FLOOR


def test_freshness_only_breaks_ties():
    # Two identical candidates, one uploaded a year earlier. At 5% weight the gap
    # must be small enough to order them without overturning any real signal.
    job = JobProfile(required_skills=["Python"])
    fresh = CandidateFacets(primary_skills=["python"], uploaded_at=NOW)
    stale = CandidateFacets(primary_skills=["python"], uploaded_at=datetime(2024, 1, 1))
    gap = _score(job, fresh).score - _score(job, stale).score
    assert 0 < gap <= 3.0


# --------------------------------------------------------------------------
# Neutral handling as a whole
# --------------------------------------------------------------------------

def test_a_silent_jd_cannot_fail_a_candidate():
    """An empty JD states no requirements, so nothing can be failed."""
    result = _score(JobProfile(), CandidateFacets(uploaded_at=NOW))
    assert all(p.neutral for p in result.parameters if p.key != "freshness")
    assert result.score >= 50.0


def test_a_stated_requirement_the_candidate_fails_still_scores_low():
    """Neutrality applies to silence, never to a requirement that was actually tested."""
    result = _score(
        JobProfile(required_skills=["Rust", "Erlang"], title="Data Scientist",
                   location="Berlin", min_years=10.0, industry="Healthcare",
                   education="PhD Physics", availability_weeks=0.0),
        CandidateFacets(primary_skills=["php"], title="Junior Designer",
                        location="Chennai", total_years=1.0,
                        experience_text="Designed posters.", education="Diploma Fine Arts",
                        availability_weeks=12.0, uploaded_at=NOW),
    )
    assert result.score < 50.0
    assert result.band == "Weak fit"


# --------------------------------------------------------------------------
# Extraction feeding the score
# --------------------------------------------------------------------------

JD_TEXT = """Senior Backend Engineer

Job Title: Senior Backend Engineer
Location: Chennai, Tamil Nadu
Industry: FinTech
Experience: 5-8 years
Education: B.E in Computer Science
Notice period: candidates who can join within 4 weeks preferred.
"""

RESUME_TEXT = """Ravi Kumar
Senior Backend Engineer
Address: Chennai, Tamil Nadu 600017
ravi@example.com | +91 98765 43210

Summary
Notice period: 2 weeks.

Skills
Python, FastAPI, PostgreSQL, Redis, Docker, Kubernets

Experience
Lead Engineer, PayFlow - built FastAPI services on PostgreSQL for a fintech payments product.

Education
B.E in Computer Science, Anna University
"""


def test_job_profile_reads_every_labelled_field():
    job = extract_job_profile(JD_TEXT, ["Python"])
    assert job.title == "Senior Backend Engineer"
    assert job.location == "Chennai, Tamil Nadu"
    assert job.industry == "FinTech"
    assert job.education == "B.E in Computer Science"
    assert (job.min_years, job.max_years) == (5.0, 8.0)
    assert job.availability_weeks == 4.0


def test_an_open_ended_experience_requirement_is_read_as_a_floor():
    job = extract_job_profile("We need 5+ years of experience.", [])
    assert job.min_years == 5.0 and job.max_years is None


def test_a_remote_job_needs_no_location():
    job = extract_job_profile("Location: Remote", [])
    assert job.is_remote and job.location is None


def test_an_unstructured_jd_yields_neutral_fields_rather_than_guesses():
    job = extract_job_profile("Come work with us. It will be fun.", [])
    assert job.location is None and job.industry is None and job.education is None
    assert job.min_years is None and job.availability_weeks is None


def test_paid_leave_is_not_read_as_a_notice_period():
    # A day count only becomes a joining window next to joining language.
    job = extract_job_profile("We offer 30 days paid leave every year.", [])
    assert job.availability_weeks is None


def test_facets_separate_claimed_skills_from_practised_ones():
    facets = extract_candidate_facets(RESUME_TEXT, title="Senior Backend Engineer",
                                      total_years=7.0, uploaded_at=NOW)
    assert "kubernets" in facets.primary_skills      # claimed, misspelt
    assert "kubernets" not in facets.technologies    # never demonstrated
    assert "fastapi" in facets.technologies          # used in Experience
    assert facets.location == "Chennai, Tamil Nadu"
    assert facets.availability_weeks == 2.0
    assert facets.education.startswith("B.E in Computer Science")


def test_a_technology_is_never_harvested_as_a_location():
    """The bug this guard exists for: a pipe-separated skills line carrying a postal
    code donated 'Java' as the candidate's city, and every location comparison then
    scored residence in a programming language."""
    resume = "Ravi\nLanguages: Java, Python, SQL | Chennai 600017\n"
    facets = extract_candidate_facets(resume)
    assert facets.location is not None
    assert "java" not in (facets.location or "").lower()


def test_declared_skills_reads_a_list_as_a_list():
    # Shape-based extraction misses plain or misspelt names; list structure does not.
    found = declared_skills("Languages: Java, Kubernets, Spring Boot\nTools: Git | Jira")
    assert {"java", "kubernets", "spring boot", "git", "jira"} <= found


def test_extraction_survives_an_empty_document():
    assert extract_job_profile("", []).title is None
    assert extract_candidate_facets("").primary_skills == []


def test_the_documented_worked_example():
    """End to end on real-shaped documents, asserting the exact published number so a
    change in any parameter has to be deliberate."""
    job = extract_job_profile(
        JD_TEXT, ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"]
    )
    facets = extract_candidate_facets(RESUME_TEXT, title="Senior Backend Engineer",
                                      total_years=7.0, uploaded_at=datetime(2026, 8, 1))
    result = _score(job, facets)

    assert _param(result, "skill").score == 93.3        # 5 exact + 1 spelling variant
    assert _param(result, "experience").score == 100.0  # 7 within 5-8
    assert _param(result, "designation").score == 100.0
    assert _param(result, "location").score == 100.0
    assert _param(result, "availability").score == 100.0
    assert result.band == "Excellent fit"


# --------------------------------------------------------------------------
# Text matching primitives
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "written,expected",
    [("React.js", "react"), ("ReactJS", "react"), ("react-js", "react"),
     ("Node.js", "node"), ("C++", "c++"), ("C#", "c#"), ("JS", "js")],
)
def test_normalisation_is_a_spelling_rule_not_a_skill_map(written, expected):
    assert tm.normalise(written) == expected


def test_normalisation_never_empties_a_term():
    assert tm.normalise("js") == "js"
    assert tm.normalise("") == ""


def test_phrase_similarity_is_symmetric():
    a, b = "Senior Backend Engineer", "Backend Engineer"
    assert tm.phrase_similarity(a, b) == tm.phrase_similarity(b, a)
