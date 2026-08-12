# Algorithm Specifications - Recruitment Reasoning

This document details the reasoning algorithms executing inside our Hiring Intelligence Platform. Every service is built to expose **Input → Reasoning → Output** flows rather than hardcoded heuristics.

---

## 1. JD & Resume Semantic Ingestion
We do not look for fixed keywords. Instead, we infer the intent of both documents structurally.

- **Input:** Raw Document (PDF or DOCX).
- **Reasoning:**
  1. Extract structural text blocks using layout-aware handlers.
  2. Parse key semantic sections (e.g., Skills, Experience, Projects).
  3. Map metadata variables (page coordinates, document origin, candidates).
- **Output:** Structured `TextNode` objects containing chunk content and coordinate metadata.

---

## 2. Requirement Extraction Taxonomy
Instead of parsing for a fixed list, the Candidate Intelligence Agent infers job parameters dynamically.

- **Input:** Parsed JD Text.
- **Reasoning:**
  1. Scan text for technical competencies using a soft tech taxonomy.
  2. Filter keywords by word boundary lookbehinds and lookaheads (supporting symbols like C++, C#, .NET).
- **Output:** List of requirement keywords dynamically targeted for evidence querying.

---

## 3. Coverage Calculation Algorithm
Calculates requirement fit metrics objectively.

- **Input:** dynamic requirements list, retrieved evidence matches list.
- **Reasoning:**
  1. For every requirement, check if matches contain relevant project details.
  2. If found, mark status = Matched. If listed as a keyword only, status = Partial. If empty, status = Missing.
  3. $\text{Coverage Score} = \left( \frac{\text{Matched} + 0.5 \times \text{Partial}}{\text{Total Requirements}} \right) \times 100$.
- **Output:** Coverage Score (0-100) and structured fit lists.

---

## 4. Evidence Confidence Algorithm
Evaluates how strongly a candidate's claims are verified by project details.

- **Input:** matched evidence text chunk.
- **Reasoning:**
  1. Measure technical specificity (presence of frameworks, libraries, adjacent APIs).
  2. Check for implementation detail depth.
  3. Validate measurable business outcomes (e.g. percentages, speed increases, revenue gains).
  4. Deduct confidence if a skill is listed as a keyword without contextual projects.
- **Output:** Confidence Score (0-100) with written logical justification.

---

## 5. Dynamic Upskilling Roadmap Algorithm
Determines realistic learning timelines by analyzing conceptual transferability.

- **Input:** Candidate demonstrated skills, absent requirements.
- **Reasoning:**
  1. Evaluate overlap between known tools and missing requirements (e.g., if Docker is known, Kubernetes complexity is scaled down).
  2. Estimate learning effort:
     $$\text{Estimated Time} = \text{Base Difficulty} - \text{Transferable Skill Credits} + \text{Conceptual Gap complexity}$$
- **Output:** Study schedules (e.g. "3-5 days") with transferable reasoning explanations.

---

## 6. Interview Question Generator
Formulates validation prompts targeting low-confidence claims.

- **Input:** Partial-status requirements, low-confidence match chunks.
- **Reasoning:**
  1. Identify claims where evidence is weak or claims are unsupported.
  2. Draft specific behavioral/technical questions to confirm actual competency (e.g. asking for configuration settings or troubleshooting scenarios).
- **Output:** Recruiter guide containing 3-5 validation questions.

---

## 7. Dynamic Rejection Email Algorithm
- **Input:** Hiring report summary, match scores, identified gaps.
- **Reasoning:**
  1. Synthesize candidate strengths to keep the email positive.
  2. Detail the exact primary gaps (missing competencies) transparently.
  3. Draft an email dynamically without using pre-written templates.
- **Output:** Warm, polite, custom-tailored candidate email draft.

---

## 8. Match Score (the headline number)
Compares a candidate against a **specific** job across nine weighted parameters.
Full specification, including every neutral value and penalty slope, in
[`Match_Score.md`](./Match_Score.md).

- **Input:** the structured job profile (`job_profile_extractor`) and the candidate
  facets (`candidate_facets_service`), both derived deterministically by Agent 1.
- **Reasoning:**
  - Score each parameter 0-100 independently, marking `neutral` any dimension the
    JD never stated.
  - Multiply by the configured weights (`settings.MATCH_WEIGHT_*`, sum 1.00):
    $$\text{Match Score} = \sum (\text{Score}_i \times W_i)$$

    | Parameter | Weight | | Parameter | Weight |
    |---|---|---|---|---|
    | Skill | 0.20 | | Education | 0.08 |
    | Experience | 0.15 | | Location | 0.08 |
    | Technology | 0.14 | | Availability | 0.07 |
    | Designation | 0.14 | | Freshness | 0.05 |
    | Industry | 0.09 | | | |

  - Clamp to 0-100, round to **one decimal**.
- **Output:** a reproducible score plus its nine-row decomposition, each row
  carrying the sentence that justifies it.
- **Invariant:** an unstated requirement scores neutrally and can never fail a
  candidate; a *stated* requirement the candidate fails still scores low.

### 8b. Evidence sub-scores (retained, no longer the headline)
Coverage, Experience Alignment, Project Relevance, Evidence Confidence and Quality
are still computed from retrieved evidence and shown separately. They are what the
strengths, weaknesses, learning roadmap and interview questions are reasoned from.
Their legacy weighted total (`settings.WEIGHT_*`) is used as `overall_score` only
for analyses evaluated before the Match Score existed.
