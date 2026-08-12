"""
Regressions found on REAL resumes (a 306-CV pool), not the synthetic test set.

Real CVs are far messier than generated ones: UUID-prefixed filenames, ALL-CAPS
section headers, company and college names in the first line, reference codes. Every
case below is an answer a recruiter actually saw and rightly called nonsense.
"""
import pytest

from app.services.ai import corpus_index_service as corpus
from app.services.ai.chat_query_understanding import QueryKind, parse_intent

from tests.chat_fixtures import make_lexicon

# A pool polluted the way the real one was: junk harvested from resume prose sits in
# the skill lists, and the body text is full of the ordinary English that produced it.
NAMES = ["Aravind Swamy", "Nithin Kumar", "Priya Sharma"]
DIRTY_SKILLS = ["AAVIN", "WITHIN", "ABOUT", "Java", "Python", "AWS"]
DIRTY_LEXICON = make_lexicon(
    NAMES,
    {n: DIRTY_SKILLS for n in NAMES},
    prose=("The work was carried out within the team and about the platform, "
           "within budget and about schedule, within scope."),
)
EMPTY_POOL = make_lexicon([], {})


# --- REGRESSION: a person's name must never be "corrected" into a skill ----
@pytest.mark.parametrize("typed,junk", [("aravind", "AAVIN"), ("nithin", "WITHIN")])
def test_names_are_not_fuzzy_matched_to_corpus_junk(typed, junk):
    """'who is aravind' searched for AAVIN — a dairy federation named in one resume."""
    intent = parse_intent(f"who is {typed}", EMPTY_POOL)
    assert junk not in intent.skills
    assert not any(c[1] == junk for c in intent.corrections)


def test_unknown_person_asks_instead_of_searching(reset_vocab=None):
    intent = parse_intent("who is aravind", EMPTY_POOL)
    assert intent.kind is not QueryKind.SKILL_SEARCH
    assert intent.skills == []


def test_possessive_skill_question_about_an_unknown_person():
    """'nithin skills' searched for WITHIN and returned a candidate named 'Skills'."""
    intent = parse_intent("nithin skills", EMPTY_POOL)
    assert "WITHIN" not in intent.skills


def test_typo_correction_still_works_for_real_technologies():
    """The fix must not disable genuine correction."""
    intent = parse_intent("who knows javaa", DIRTY_LEXICON)
    assert [s.lower() for s in intent.skills] == ["java"]
    assert intent.corrections and intent.corrections[0][0] == "javaa"


def test_a_known_person_is_still_found():
    intent = parse_intent("who is Aravind Swamy", DIRTY_LEXICON)
    assert intent.named_candidates == ["Aravind Swamy"]


def test_real_skill_search_is_unaffected():
    intent = parse_intent("who has java and aws experience", DIRTY_LEXICON)
    assert {s.lower() for s in intent.skills} == {"java", "aws"}


# --- REGRESSION: vocabulary hygiene ---------------------------------------
@pytest.mark.parametrize("junk", [
    "WITHIN", "ABOUT", "ACHIEVEMENTS", "ACADEMIC", "SUMMARY", "EXPERIENCE",
    "CONFIDENTIAL", "CURRICULUM", "0AFZZ", "8U04", "234J", "50CGPA", "2017",
])
def test_boilerplate_and_ids_are_not_skills(junk):
    assert not corpus._is_plausible_skill_token(junk)


@pytest.mark.parametrize("real", ["Java", "AWS", "k8s", "S3", "CI/CD", "PostgreSQL", "ABAP"])
def test_real_technologies_survive_the_filter(real):
    assert corpus._is_plausible_skill_token(real)


# --- REGRESSION: candidate names --------------------------------------------
@pytest.mark.parametrize("heading", [
    "EXPERIENCE", "SUMMARY", "PROFESSIONAL SUMMARY", "CURRICULUM VITAE",
    "RESUME", "EDUCATION", "Skills", "CONFIDENTIAL", "About Me",
])
def test_section_headings_are_rejected_as_names(heading):
    assert corpus._clean_person_name(heading) is None


@pytest.mark.parametrize("org", [
    "Ethiraj College for Women", "Periyar University", "Infosys Limited",
    "Wipro Technologies", "Acme Solutions",
])
def test_institutions_are_rejected_as_names(org):
    assert corpus._clean_person_name(org) is None


@pytest.mark.parametrize("name", ["Aravind Swamy", "Priya", "Md Salahuddin", "Patricia V. Rose"])
def test_real_names_are_kept(name):
    assert corpus._clean_person_name(name) == name


@pytest.mark.parametrize("filename,expected", [
    ("00ca930e-1d49-47c5-a845-a1d475aeb237-Poravi_Resume.docx", "Poravi"),
    ("00de11d61c7a4b5f9e8af6c98dd81c35-Sunitha Dasari_Resume.docx", "Sunitha Dasari"),
    ("0a9cc385434e44ac9761f87032fb8125-Santhosh Resume1 (1).docx", "Santhosh"),
])
def test_upload_id_prefixes_are_stripped_from_filenames(filename, expected):
    """Candidate cards showed raw UUIDs because the filename stem was used verbatim."""
    assert corpus._name_from_filename(filename) == expected


def test_unusable_filename_does_not_become_a_name():
    assert corpus._name_from_filename("00b1ef7ba3ff4fbcb9f159172fe1016a-9757e889.docx") is None
