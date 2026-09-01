# System prompts

Eight prompts, one file each, verbatim as they run today. Nothing else — no C#,
no wiring.

A system prompt alone is half a contract. Each section below gives the other
half: **what to put in the user message**, **what comes back**, and **the call
settings**, because several of these break quietly if you get those wrong.

Every one currently runs on `llama3.1:8b` via Ollama's `/api/chat`.

| # | File | Job | JSON out |
|---|---|---|---|
| 1 | `1-evaluator.system.txt` | Judge one CV against one JD | yes |
| 2 | `2-turn-router.system.txt` | Route a message to one action | yes |
| 3 | `3-answer-composer.system.txt` | Word an answer over given facts | no |
| 4 | `4-skill-transfer.system.txt` | Does an owned skill transfer? | yes |
| 5 | `5-jd-extraction.system.txt` | JD text → structured fields | yes |
| 6 | `6-cv-extraction.system.txt` | CV text → structured fields | yes |
| 7 | `7-shortlist-advice.system.txt` | Advise over scored results | no |
| 8 | `8-cover-letter.system.txt` | Write an application letter | no |
| 9 | `9-evidence-verify.system.txt` | Does this passage evidence this skill? | yes |
| 10 | `10-rerank.system.txt` | Rate evidence against a request, 0-100 | yes |

Prompts 1–8 come from the .NET portal. **9 and 10 are the retrieval-stage
prompts**, from the Python platform's `chat_llm_retrieval.py`, and they are the
only two in this set that change *which* candidates surface and in what order.

---

## The rule they all share

**The model never supplies a fact.** Each prompt is handed evidence and asked to
judge or phrase it. None is asked what it knows. If you carry these across,
carry that with them — every constraint in them exists because it was broken
once and something plausible and wrong reached a person.

Two consequences worth building for:

- **Every one of these must be allowed to fail.** Have a deterministic answer
  ready for each, and use it when the model is absent, slow or malformed. None
  of these prompts is on a path that may hang.
- **Verify quotes.** Prompts 1 and 4 ask the model to copy strings from a source
  document. Check them. Measured on an 8B, roughly every quote in prompt 1 was
  invented — the right skills, justified with fabricated sentences.

---

## 1. Evaluator — `1-evaluator.system.txt`

The main one. Everything else is supporting cast.

**User message:** the JD then the CV, clearly separated and labelled. Section
headings in the CV must survive — they are the entire basis of the
STRONG/WEAK/MISSING distinction, so a pre-parsed skill list instead of the
document throws away the thing being judged.

```
=== JOB DESCRIPTION ===
Title: ...
Seniority: ...
Minimum years: ...
Required: ...
Preferred (nice to have): ...
Stated qualifications:
- ...
Responsibilities:
- ...

Full posting text:
<raw JD, clipped to ~5000 chars>

=== CANDIDATE RÉSUMÉ (verbatim, section headings intact) ===
<raw CV, clipped to ~9000 chars>
```

**Returns:** `requirements_evaluation[]` (each with `jd_requirement`, `kind`,
`match_level`, `resume_quote`, `reasoning`), `weights[]`, `knockout_indicators`,
`justification_summary`, `alternate_role`.

**Note it returns no score.** That is deliberate and is the single most
important thing to preserve. Ask an 8B for a percentage and the same CV and JD
return 41 and 67 on consecutive calls. Compute the number from the rows:

```
STRONG = 1.0, WEAK = 0.5, MISSING = 0.0
score  = Σ over kinds: weight(kind) × mean(credit of that kind's rows)
```

Score each kind against **its own** rows, not the whole list, or one must-have
gets buried under three core skills that happen to be present.

Then four corrections that matter more than the arithmetic:

- **Cap `responsibility` at 15% of the weighting** and `nice_to_have` at 10%,
  redistributing the surplus. Responsibility bullets describe the job on offer,
  not a checklist the applicant must have ticked, and there are always more of
  them. Uncapped they dominate by count and mark down everyone who can do the
  job. This was worth ~20 points on a real candidate.
- **Derive knockouts from the rows**, not from `knockout_indicators` alone. A
  MISSING row of kind `must_have` / `experience` / `education` fires its
  indicator regardless of what the model reported — trusting the booleans let a
  candidate with neither the required tool nor the required years score 45%. Let
  the model's flags *add* a knockout, never remove one.
- **A knockout caps the total at 34.** Not 35 — the bands put Weak Match at 35+,
  so a ceiling of 35 puts a disqualified candidate in the same band as a merely
  poor one.
- **Compose the summary yourself** from the rows. Asked for a free paragraph,
  the model wrote that a candidate was a "wireframing and user research expert"
  — the two things its own rows had marked MISSING. A summary is what gets read
  instead of the evidence, so it is the worst place for an invention, and unlike
  a quote there is nothing to check it against. Keep the per-row `reasoning`:
  that one sits beside its own verdict, where a false claim contradicts itself.

**Years are extracted, never compared.** The prompt forbids the model from
deciding whether a duration meets a minimum, and the schema has no years
dealbreaker — only `candidate_years_experience`, a number. Asked whether 2.6
years met a 2-year minimum, an 8B said no and capped a candidate who *exceeded*
the requirement. Do the subtraction in code, and fire nothing when either side
is unknown: "the résumé does not say" is not evidence of a shortfall.

**Cross-check the posting's own required-skills list.** Do not key the
mandatory-skill dealbreaker solely on the `kind` the model assigned. A tool the
posting explicitly requires, absent from the CV, labelled `core_skill` instead
of `must_have`, costs nothing — a hole exactly the width of one
misclassification. The posting already told you which skills are mandatory; let
that list decide, and let the model's label only add to it.

**Call settings:** `format: json`, **`temperature: 0`**, and cache the result per
(CV, JD, prompt version). A judgement that changes between two identical asks is
not a judgement.

**Grounding order matters.** Verify quotes *before* computing the score, and
correct the row when a quote cannot be found — an unverifiable claim becomes
MISSING while it can still change the number. Checking afterwards leaves a
percentage standing on evidence you then threw away.

**What grounding must not do** is require the *requirement's own words* to
appear in the CV. That check re-imposes keyword matching at the last step and
discards correct readings — "this Angular work evidences component-based
frontend development" was thrown away because that phrase was not written in the
document. Verify the quote; never gate on the requirement.

---

## 2. Turn router — `2-turn-router.system.txt`

**User message:** the candidate's message, plus a CONTEXT block listing the jobs
currently on screen as `id — title — company`, the current filters, and a few
turns of history.

**Returns:** `{action, job_ids, filters, reset_filters, undo_filters, question, clarify}`.

**Validate everything before acting on it.** Every `job_ids` entry must exist;
every filter value must be one your system recognises. A plan referring to
something unreal is discarded and the deterministic parse used instead. Keep a
flag recording which produced the plan.

`format: json`.

---

## 3. Answer composer — `3-answer-composer.system.txt`

**User message:** `FACTS:\n<the facts>\n\nQUESTION: <question>`.

The facts must be complete — this prompt has no other source. It phrases; it
does not decide. Pair it with a deterministic sentence and fall back to that
whenever the model is unavailable or its answer fails screening.

Worth screening the output for numbers that do not appear in the facts.

---

## 4. Skill transfer — `4-skill-transfer.system.txt`

**User message:** the requirement list and the candidate's skill list, both
verbatim.

**Returns:** `{transfers: [{requirement, candidate_skill, reason}]}`.

Both `requirement` and `candidate_skill` must be checked back against the lists
you supplied. An empty list is a correct answer and the prompt says so — without
that line the model finds a transfer for everything, because everything is
"technology".

`format: json`. Cache per (job, skill-set).

---

## 5 & 6. Extraction — `5-jd-extraction.system.txt`, `6-cv-extraction.system.txt`

**User message:** the raw text, *plus what your deterministic parser already
read*:

```
RAW JOB POSTING
---------------
<raw text, clipped to ~6000 chars>

WHAT THE PARSER ALREADY READ
----------------------------
title: ...
company: ...
required_skills: ...
```

Showing the parser's output turns this from extraction into correction, which is
a much easier task and produces far fewer inventions. Keep the deterministic
parse as the result if the model fails.

These two run **upstream of the evaluator**. If a card's requirement list looks
wrong, check here first — the evaluator can only atomise what extraction handed
it.

`format: json`.

---

## 7. Shortlist advice — `7-shortlist-advice.system.txt`

**User message:** the scored roles with their percentages, and the missing
skills ranked by *how many of those roles each one blocks*. The prompt tells the
model to preserve that order, so the ranking must already be correct — the model
is not deciding what to learn first, it is reading out a computed answer.

Streamed to the UI. Fallback is the same advice, composed.

---

## 8. Cover letter — `8-cover-letter.system.txt`

**User message:**

```
ROLE: ...
COMPANY: ...
CANDIDATE'S CURRENT TITLE: ...
YEARS OF EXPERIENCE: ...
EVIDENCED SKILLS: <only skills verified in the CV>
ASSESSED STRENGTHS: <up to 3>
```

**Withhold the role's other requirements entirely.** Listing them and forbidding
their mention is an instruction small models do not follow — measured here, the
model named a technology straight out of the list it had just been told to
avoid, and the letter was discarded. Removing the temptation beats repeating the
prohibition. That is a general lesson for all eight: constrain by what you put
in the context, not by what you tell it to ignore.

Screen the output for any skill outside EVIDENCED SKILLS and discard the letter
if one appears.

---

## 9 & 10. Retrieval — `9-evidence-verify.system.txt`, `10-rerank.system.txt`

These sit **on top of** dense + sparse retrieval, never in place of it. They
only ever judge text that deterministic retrieval already returned.

**9 — evidence verification.** User message is a numbered list of
`(id, skill, passage)` triples. Returns `[{id, verdict: "yes|weak|no"}]`. It
replaces a string check that asked whether the skill's own words appeared in the
passage — blunt in both directions: *"led the sprint ceremonies and unblocked
the team"* evidences Communication without containing the word, while
"communication" in a skills list is a claim, not proof.

**10 — re-ranking.** User message is `Recruiter request: …` then the candidates
with their evidence. Returns `[{id, score}]`. It reads what a passage *says*,
which a weighted sum of coverage and depth never could.

**The containment is the whole design.** Every id shown to the model came out of
the vector index, and every id it returns is checked back against the set it was
given. A wrong verdict can reorder or demote a **real** candidate; it can never
invent one, and it can never resurrect one the funnel excluded. That property is
what makes both stages safe to default **off** — with the model absent, slow or
malformed, the deterministic ranking stands exactly as it was. Every failure
path returns "no opinion", never an exception.

Batch these. A sequential per-item loop cost ~13 serial round trips in the .NET
portal for one shortlist; send many items per call instead.

---

## Porting to a different model

- **Recheck prompt 1 first.** The responsibility and nice-to-have ceilings, and
  the quote-verification fallbacks, are compensations for a small model. A
  stronger one may not need them, and they will hold it back if it doesn't.
- **Keep `temperature: 0` on 1, 4, 5 and 6.** These produce structured
  judgements, not prose.
- **`format: json` is not a guarantee.** Recover the object between the first
  `{` and the last `}` — small models still wrap JSON in prose or a fence often
  enough to be worth handling rather than discarding a whole evaluation over.
- **Watch the prompt budget.** Prompt 1's system text is ~9.5k characters before
  the JD and CV are added. On a small model with a large context that is fine;
  on a constrained deployment, the worked examples are the first thing to trim.
- **Version the prompts.** Cache keys must include the version, or you serve
  judgements produced by instructions that no longer exist. Prompt 1 currently
  runs as `eval-v4`.
