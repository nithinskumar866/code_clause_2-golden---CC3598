# Match Score — the specification

The number the platform reports for a candidate against a job. Deterministic,
0–100, one decimal.

**The AI does not decide it.** The AI's job finishes upstream: it extracts structure
from the documents — skills, sections, titles, evidence. Once that structure exists,
the score is arithmetic. The same candidate against the same job produces the same
number, on any machine, in any run, for anyone implementing this specification.

---

## The formula

```
Match Score = Skill        × 20%
            + Experience   × 15%
            + Technology   × 14%
            + Designation  × 14%
            + Industry     ×  9%
            + Education    ×  8%
            + Location     ×  8%
            + Availability ×  7%
            + Freshness    ×  5%
```

Weights sum to 1.00 and live in `backend/app/core/config.py` as `MATCH_WEIGHT_*`.
The result is clamped to 0–100 and rounded to one decimal place.

## Interpretation

| Range | Band |
|---|---|
| 85–100 | Excellent fit |
| 70–84 | Strong fit |
| 50–69 | Moderate fit |
| below 50 | Weak fit |

These bands are the single source of truth for every threshold in the product: the
dashboard's Selected / Borderline / Rejected buckets (≥ 70 / ≥ 50 / below), the
recruiter recommendation wording, the score colours, and whether a rejection draft
is written. A dashboard that calls someone "Rejected" while their report reads
"Strong fit" would be two opinions presented as one fact.

---

## The two rules that govern everything

### 1. Silence is neutral, not failure

When a job description does not state a location, an industry, an education bar or
a joining window, that parameter scores mid-range and is flagged `neutral`.

A requirement nobody wrote down must never be able to sink a candidate. Without
this rule the score would measure how verbose the JD is rather than how good the
candidate is, and a terse JD would rank everyone as a poor fit.

Failing a **stated** requirement is a different thing entirely and still scores low.
A candidate in the wrong city scores 20 on Location; a candidate whose city is
unknown scores 50. The two must not be conflated.

**Consequence worth knowing:** Industry and Availability are absent from most real
resumes and most real JDs. Expect them to sit at their neutral values often, which
means roughly 16% of the score is a constant offset and the effective spread between
candidates is narrower than 0–100. The UI says so directly when it happens, and the
fix is a more specific JD, not a different formula.

### 2. Claimed and practised are different parameters

**Skill Match** reads the whole resume, including the skills list — a claim.
**Technology Match** reads only Experience and Projects — evidence.

A candidate who lists Kubernetes but never used it scores full marks on the first
and zero on the second. That gap is precisely the judgement a recruiter is trying to
make, and collapsing the two into one number destroys it.

---

## The nine parameters

### 1. Skill Match — 20%
Required skills against every skill the resume mentions. Credit per requirement:

| Credit | Case |
|---|---|
| 1.0 | the same skill once spelling conventions are normalised (`ReactJS`, `React.js`, `react` are one skill) |
| 1.0 | a semantically equivalent skill under another name (JS ↔ JavaScript) |
| 0.6 | a near-miss that is a typo rather than a different skill (`Kubernets`) |
| 0.0 | absent |

Score is the total credit over the number of requirements.

**Aliases are resolved algorithmically, never from a table.** Normalisation is a
rule about orthography — the same rule applies to a framework released tomorrow.
Genuine synonyms go through `skill_semantics_service`, which decides equivalence by
embedding similarity. This keeps Golden Rule 4 intact: no hardcoded skill mappings.

The typo tier requires at least four characters. `go` and `god` sit above any
sensible character-similarity threshold while naming completely different things.

### 2. Experience Match — 15%
Inside the JD's range → 100. Outside → **15 points per year** away from the nearest
edge. An open-ended requirement (`5+ years`) has no upper penalty. Neutral when
either side is unknown.

### 3. Technology Match — 14%
Same credit algorithm as Skill Match, but the pool is only what appears in Experience
and Projects.

### 4. Designation Match — 14%
Similarity between the JD's title and the candidate's current role. Token
containment, so `Senior Backend Engineer` fully matches `Backend Engineer` —
seniority adjectives and company decoration are not a mismatch. Neutral when either
side has no title.

### 5. Industry Match — 9%
The JD's stated domain against the candidate's work history text.

### 6. Education Match — 8%
The JD's stated qualification against the resume's Education section.

### 7. Location Match — 8%
| Score | Case |
|---|---|
| 100 | remote role, or same location |
| 20–100 | partial, scaled by similarity, for a nearby or overlapping location |
| 20 | a different location — reachable by relocation, so not zero |
| 50 | either side did not state a location |

### 8. Availability Match — 7%
Within the employer's window → 100. Otherwise **10 points per week late**. If the JD
states no deadline → neutral 50. If the JD asked and the resume is silent → 40: the
employer asked and the candidate did not answer, which is a small negative signal
rather than an absence of information.

### 9. Resume Freshness — 5%
100 on the day of upload, decaying linearly to a floor of 40 over one year. At 5%
weight this separates otherwise-identical candidates by at most about 3 points — a
tie-breaker, not a signal.

The upload timestamp is carried through the workflow rather than read from the clock
at scoring time, so re-running an analysis reproduces its original score.

---

## Where it lives

| Concern | Module |
|---|---|
| The nine parameters and the formula | `services/ai/match_scoring_service.py` |
| Structured JD facts (title, location, industry, education, years, notice) | `services/ai/job_profile_extractor.py` |
| Resume facets (claimed vs practised skills, location, education, notice) | `services/ai/candidate_facets_service.py` |
| String primitives (normalisation, typo detection, phrase similarity) | `services/ai/text_matching.py` |
| Weights, neutral values, penalty slopes, bands | `core/config.py` (`MATCH_*`) |
| API contract | `schemas/analysis.py` :: `MatchScore`, `MatchParameter` |
| UI | `components/analysis/MatchScoreBreakdown.tsx` |

Extraction runs in **Agent 1** (Candidate Intelligence) because it is evidence
gathering. Scoring runs in **Agent 2** (Hiring Decision) because it is a judgement.
Agent 2 never opens a document; it only reads the evidence package. The two-agent
rule is unchanged.

---

## What did *not* change

The five legacy sub-scores — requirement coverage, experience alignment, project
relevance, evidence confidence, resume quality — are still computed and still shown,
under "Evidence Sub-scores". They no longer decide the headline number, but they are
what the strengths, weaknesses, learning roadmap and interview questions are reasoned
from, so removing them would gut the written report.

AI Rating (resume quality) remains a separate, job-independent signal and is not part
of the Match Score.

Analyses evaluated before this specification carry no structured facets. Those fall
back to the legacy weighted total and report `match_score: null` — scoring them as if
every dimension were neutral would produce a confident-looking ~50 that means nothing.

---

## Reproducing a score by hand

Every parameter reports `score`, `weight`, `contribution` and a `basis` sentence.
Multiply, add, round to one decimal. The worked example asserted in
`tests/test_match_scoring.py::test_the_documented_worked_example`:

```
Skill Match          93.3 x 0.20 = 18.7   5 exact matches + 1 spelling variant
Experience Match    100.0 x 0.15 = 15.0   7 years inside the required 5-8
Technology Match     50.0 x 0.14 =  7.0   3 of 6 shown in real work
Designation Match   100.0 x 0.14 = 14.0
Industry Match      100.0 x 0.09 =  9.0
Education Match     100.0 x 0.08 =  8.0
Location Match      100.0 x 0.08 =  8.0
Availability Match  100.0 x 0.07 =  7.0
Resume Freshness     99.2 x 0.05 =  5.0
                                   -----
Match Score                         91.6   Excellent fit
```
