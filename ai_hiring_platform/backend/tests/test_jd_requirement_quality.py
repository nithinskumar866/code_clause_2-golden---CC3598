"""
Requirement-extraction quality tests.

Every case here comes from a real System Administrator JD that produced 36
"requirements" where ~12 exist. The junk was not cosmetic: unmatchable phrases like
"seeking" and "Contribute" counted as MISSING must-haves and dragged a genuinely
strong candidate down to 51%, and five separate Java entries diluted coverage while
turning the report into a wall of near-identical rows.
"""
import pytest

from app.services.ai.jd_requirement_extractor import extract_requirements

JD = """
Job Summary
We are seeking an experienced System Administrator with strong expertise in server administration, Java-based applications, infrastructure management, process optimization, database migrations, Python automation, and AI tool integration.
Key Responsibilities
Administer and maintain enterprise server environments across production systems.
Monitor system availability, performance, capacity requirements, and infrastructure health.
Support and maintain Java-based enterprise applications and related deployment activities.
Contribute to initiatives that improve operational efficiency and service quality.
Required Skills and Experience
7+ years of experience in System Administration, Infrastructure Operations, or a related role.
Hands-on experience managing server environments, capacity requirements, and deployment queries.
Working knowledge of Java and experience supporting Java-based technical solutions.
Experience with operational workflows and process optimization.
Proficiency in Python for scripting, automation, or operational support.
Preferred Skills
Cloud platform exposure such as AWS, Azure, or Google Cloud.
Knowledge of networking, security, and high-availability environments.
Ideal Candidate Profile
The ideal candidate combines strong system administration experience with practical knowledge of Java, Python, database migrations, and AI-assisted operational workflows.
Experience | Location | Employment Type
7-10 Years | Chennai, Tamil Nadu | Full-Time
"""


@pytest.fixture(scope="module")
def reqs():
    return extract_requirements(JD)


@pytest.fixture(scope="module")
def lowered(reqs):
    return [r.lower() for r in reqs]


@pytest.mark.parametrize("junk", ["seeking", "contribute", "monitor", "hands-on",
                                  "location", "employment type", "technology"])
def test_prose_verbs_and_boilerplate_are_not_requirements(lowered, junk):
    """Sentence-initial duty verbs and posting metadata are grammar, not competencies."""
    assert junk not in lowered


def test_no_phrase_spans_a_sentence_boundary(reqs):
    """A requirement must come from one sentence; '<object>. <cue> <object>' is a bug."""
    for r in reqs:
        assert "." not in r.rstrip("."), f"{r!r} straddles a sentence boundary"


def test_java_variants_collapse_to_one_requirement(lowered):
    """java / Java-based applications / practical knowledge of Java are ONE ask."""
    java_like = [r for r in lowered if "java" in r]
    assert len(java_like) == 1, f"expected one Java requirement, got {java_like}"


def test_trailing_punctuation_does_not_duplicate_a_skill(lowered):
    """'Google Cloud.' and 'google cloud' must not both be requirements."""
    normalised = [r.rstrip(".") for r in lowered]
    assert len(normalised) == len(set(normalised))


def test_qualified_variants_collapse(lowered):
    """'AI' and 'AI-assisted' are the same requirement."""
    assert len([r for r in lowered if r.startswith("ai")]) <= 1


def test_lowest_stated_experience_bar_is_used(lowered):
    """The JD asks for 7+; scoring against the range's upper bound punishes everyone."""
    years = [r for r in lowered if "years experience" in r]
    assert years == ["7+ years experience"]


def test_real_requirements_survive(lowered):
    """Cleaning must not throw away the actual asks."""
    for expected in ["java", "python", "server administration", "database migrations",
                     "process optimization", "infrastructure management"]:
        assert expected in lowered, f"lost real requirement {expected!r}"


def test_extraction_stays_concise(reqs):
    """~12-22 real requirements; 36 was the symptom that started this."""
    assert len(reqs) <= 24, f"{len(reqs)} requirements is a wall of text: {reqs}"


def test_extraction_is_deterministic():
    assert extract_requirements(JD) == extract_requirements(JD)
