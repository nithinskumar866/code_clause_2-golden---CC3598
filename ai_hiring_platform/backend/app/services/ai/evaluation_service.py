import os
import re
import json
from typing import List, Dict, Any
from pydantic import ValidationError
from app.core.config import settings
from app.core.constants import LEARNING_TRANSFER_BANDS, PARTIAL_SKILL_LEARNING_TIME
from app.core.logging import logger
from app.schemas.analysis import HiringReport, RequirementFit, LearningRoadmapItem
from app.services.ai import skill_semantics_service
from app.services.ai import authenticity_service
from app.services.ai import llm_service
from app.services.ai.retrieval_service import _KNOWN_TECH


def _short(text: str, limit: int = 90) -> str:
    """One-line, trimmed snippet of evidence for readable explanations."""
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*")


def _candidate_tokens(chunk: str) -> set:
    """Extract whole-word skill tokens from a chunk (never sub-spans of a word, so
    'PyTorch' stays 'pytorch' rather than being split into 'pyt'). Keeps known-tech
    taxonomy terms plus full words that look technical (CamelCase / acronym / carries
    a digit or tech punctuation)."""
    lowered = chunk.lower()
    found = {t for t in _KNOWN_TECH if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", lowered)}
    for m in _WORD_RE.finditer(chunk):
        core = m.group(0).strip("+#/.")
        if len(core) < 2:
            continue
        techy = (
            any(ch.isdigit() for ch in core)
            or any(ch in "+#/." for ch in core)
            or core.isupper()                 # acronym: SQL, AWS, API
            or core[1:] != core[1:].lower()   # internal caps: PyTorch, FastAPI
        )
        if techy:
            found.add(core.lower())
    return found


def _extract_candidate_skills(retrieval_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Recover the candidate's actual skill vocabulary from the retrieved evidence
    chunks, so transfer/equivalence reasoning can compare a requirement against
    what the candidate really has (not just against the JD's own terms).

    Returns {skill_lower: {display, narrative, chunk, section}} where `narrative`
    marks a skill shown in an Experience/Projects section (real, hands-on) rather
    than merely listed. Skills are extracted generically, never from a hardcoded list.
    """
    skills: Dict[str, Dict[str, Any]] = {}
    for item in retrieval_results:
        for m in item.get("matches", []):
            chunk = m.get("chunk", "") or ""
            section = m.get("section", "") or ""
            narrative = section in ("Experience", "Projects")
            for tok in _candidate_tokens(chunk):
                cur = skills.get(tok)
                # Prefer a narrative occurrence (stronger evidence) if one exists.
                if cur is None or (narrative and not cur["narrative"]):
                    skills[tok] = {"display": tok, "narrative": narrative, "chunk": chunk, "section": section}
    return skills

def _weighted_coverage(retrieval_results: List[Dict[str, Any]]) -> int:
    """Requirement coverage weighted by importance: missing a must-have costs more than
    missing a nice-to-have. Falls back to equal weights when priority is absent, so
    externally-supplied evidence (e.g. tests) scores exactly as before."""
    total = sum(it.get("weight", settings.REQUIREMENT_WEIGHT_MUST) for it in retrieval_results)
    matched = sum(
        it.get("weight", settings.REQUIREMENT_WEIGHT_MUST)
        for it in retrieval_results if it.get("matches")
    )
    return int((matched / total) * 100) if total > 0 else 0


def evaluate_evidence(evidence_data: Dict[str, Any]) -> HiringReport:
    """
    Algorithmic evaluation pipeline. Invokes LLM reasoning or falls back to
    a deterministic rule-based mock engine if API credentials are not found.
    """
    llm = llm_service.get_llm()
    if not llm:
        logger.warning("No LLM configured. Falling back to deterministic reasoning engine.")
        return run_mock_evaluation(evidence_data)
        
    try:
        prompt = _build_reasoning_prompt(evidence_data)
        response = llm.complete(prompt)
        report_data = _extract_json(response.text)
        return _finalize_llm_report(report_data, evidence_data)
    except Exception as e:
        logger.error(f"LLM reasoning failed: {e}. Falling back to deterministic engine.", exc_info=True)
        return run_mock_evaluation(evidence_data)

# --- Provider-agnostic LLM reasoning (grounded in retrieved evidence) ----------
# The LLM owns judgment (per-requirement verdicts, narrative, recommendation); the
# deterministic algorithms own the reproducible numbers (coverage, quality,
# authenticity, overall). The LLM only ever sees retrieved evidence — never the raw
# resume — so Golden Rule 2 (no raw-doc LLM access) holds.

_REASONING_SCHEMA = """{
  "experience_score": 0,        // 0-100, from Experience-section evidence depth
  "project_score": 0,           // 0-100, from Projects evidence
  "confidence_score": 0,        // 0-100, overall evidence strength
  "summary": "",
  "requirements": [
    {"requirement": "", "category": "", "status": "Matched|Partial|Missing",
     "matched_evidence": "", "explanation": "", "limitations": "", "confidence": 0}
  ],
  "strengths": [], "weaknesses": [], "skill_relationships": [],
  "missing_skills": [],
  "learning_roadmap": [{"skill": "", "estimated_time": "", "reason": ""}],
  "interview_questions": [],
  "recruiter_recommendation": "",
  "rejection_email": null
}"""


def _format_evidence_for_prompt(evidence_data: Dict[str, Any]) -> str:
    """Render the evidence package as a grounded, per-requirement view — the ONLY
    candidate information the LLM sees."""
    lines: List[str] = []
    for it in evidence_data.get("retrieval_results", []):
        req = it.get("requirement", "")
        imp = it.get("importance", "must")
        matches = it.get("matches", [])
        lines.append(f'- Requirement: "{req}"  [{imp}-have]')
        if not matches:
            lines.append("    Evidence: (none retrieved from the resume)")
        for m in matches[:3]:
            sec = m.get("section", "Unknown")
            txt = " ".join((m.get("chunk", "") or "").split())
            lines.append(f'    Evidence [{sec}]: "{txt}"')
    return "\n".join(lines) if lines else "(no requirements extracted)"


def _build_reasoning_prompt(evidence_data: Dict[str, Any]) -> str:
    """Grounded recruiter-reasoning prompt. Provider-agnostic plain text."""
    return (
        "You are a senior technical recruiter evaluating one candidate against a "
        "job's requirements. Reason like an expert who understands how technologies "
        "relate — not a keyword matcher.\n\n"
        "For each requirement you get its importance (must vs nice-to-have) and the exact "
        "evidence retrieved from the resume. Reason ONLY from this evidence — never invent "
        "a skill that is not shown.\n\n"
        f"EVIDENCE:\n{_format_evidence_for_prompt(evidence_data)}\n\n"
        "Return a SINGLE valid JSON object (no markdown, no commentary) with EXACTLY "
        f"this shape:\n{_REASONING_SCHEMA}\n\n"
        "How to reason (this is the important part — apply it to ANY technology, do not "
        "rely on a fixed list of known tools):\n"
        "- Use your knowledge of how technologies relate — families, ecosystems, "
        "programming languages, frameworks, databases, cloud platforms, DevOps tools, "
        "AI/ML libraries, and front-end/back-end stacks — to judge each requirement from "
        "the evidence, not by literal keyword match.\n"
        "- EQUIVALENT skills satisfy a requirement. If the evidence shows a skill that is "
        "interchangeable with the requirement in real work, the requirement is Matched or "
        "Partial, NOT Missing.\n"
        "- IMPLIED skills count. If demonstrated work necessarily requires the requirement "
        "(e.g. a project built on a framework implies its underlying language), the "
        "evidence for one supports the other.\n"
        "- TRANSFERABLE skills ease a gap. When the candidate lacks the exact requirement "
        "but has a closely-adjacent skill in the same family, keep the status honest "
        "(usually Missing or Partial) but say in the explanation that it is a short "
        "ramp-up because of that related experience — name the specific related skill you "
        "saw in the evidence.\n"
        "- Judge DEPTH from evidence type. A skill shown only in a skills list, with no "
        "project or outcome, is Partial and the limitation should say depth is unconfirmed. "
        "A skill with genuine project/experience evidence is Matched. A skill with no "
        "evidence and nothing related in the same family is Missing. Never accept a bare "
        "keyword as proof, and never invent skills or evidence not present in the text.\n"
        "- status: Matched (clearly shown, equivalent, or implied), Partial "
        "(mentioned/thin), Missing (no evidence and nothing transferable).\n"
        "- explanation (relevance) and limitations: PLAIN, simple English a busy recruiter "
        "reads at a glance. State clearly what is strong and what is the gap — the why and "
        "the what, grounded in the specific evidence. No jargon, no filler.\n"
        "- matched_evidence: quote the specific evidence you used (empty if Missing).\n"
        "- confidence (0-100): how strongly the evidence supports your conclusion.\n"
        "- interview_questions: make them USEFUL to the interviewer. For a transferable "
        "gap, ask a question that reveals whether the candidate can truly cross over from "
        "their related experience to the requirement. For a shallow claim, ask for one "
        "concrete example and its outcome. For a strong skill, probe depth.\n"
        "- Weight must-haves more heavily than nice-to-haves in strengths, weaknesses, and "
        "the recommendation.\n"
        "- rejection_email: a brief, kind draft ONLY if the candidate is clearly weak "
        "overall; otherwise null."
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating markdown fences and
    surrounding prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(t[start:end + 1])
        raise


def _finalize_llm_report(report_data: Dict[str, Any], evidence_data: Dict[str, Any]) -> HiringReport:
    """Overlay deterministic ownership on the LLM's reasoning: quality + authenticity
    + weighted coverage + per-requirement importance are computed in code
    (reproducible), then overall is recombined from the configured weights."""
    results = evidence_data.get("retrieval_results", [])

    auth = authenticity_service.assess_authenticity(evidence_data)
    report_data["quality_score"] = auth.quality_score
    report_data["authenticity"] = auth.assessment.model_dump()
    report_data["coverage_score"] = _weighted_coverage(results)
    # Candidate profile is computed deterministically upstream (Agent 1); carry it through.
    report_data["candidate_profile"] = evidence_data.get("candidate_profile")

    # Stamp requirement importance/weight from evidence onto the LLM's fits.
    priority_by_req = {str(it.get("requirement", "")).lower(): it for it in results}
    for r in report_data.get("requirements", []):
        ev = priority_by_req.get(str(r.get("requirement", "")).lower())
        if ev:
            r["importance"] = ev.get("importance", "must")
            r["weight"] = ev.get("weight", settings.REQUIREMENT_WEIGHT_MUST)

    cov = report_data.get("coverage_score", 0)
    exp = int(report_data.get("experience_score", 0) or 0)
    proj = int(report_data.get("project_score", 0) or 0)
    conf = int(report_data.get("confidence_score", 0) or 0)
    qual = auth.quality_score
    overall = int(
        (cov * settings.WEIGHT_COVERAGE) +
        (exp * settings.WEIGHT_EXPERIENCE) +
        (proj * settings.WEIGHT_PROJECTS) +
        (conf * settings.WEIGHT_CONFIDENCE) +
        (qual * settings.WEIGHT_QUALITY)
    )
    report_data["experience_score"] = exp
    report_data["project_score"] = proj
    report_data["confidence_score"] = conf
    report_data["overall_score"] = min(max(overall, 0), 100)

    return HiringReport.model_validate(report_data)


def _select_learning_band(transfer_score: float):
    """Return (estimated_time, strength_label) for a missing skill's transfer score."""
    for threshold, est_time, strength in LEARNING_TRANSFER_BANDS:
        if transfer_score >= threshold:
            return est_time, strength
    # Bands always include a 0.0 floor; this is a defensive fallback.
    return LEARNING_TRANSFER_BANDS[-1][1], LEARNING_TRANSFER_BANDS[-1][2]


def run_mock_evaluation(evidence_data: Dict[str, Any]) -> HiringReport:
    """
    Deterministic reasoning engine (used when no LLM is configured). Beyond direct
    keyword-to-evidence matching it reasons about the candidate's real skill set:
      * equivalent skills rescue a Missing requirement (SQL satisfies MySQL);
      * closely-related skills are noted as transferable (Django eases FastAPI),
        without falsely claiming the skill;
    and writes plain-language relevance/limitations that state what is strong and
    what is the gap. Interview questions are transfer-aware.
    """
    retrieval_results = evidence_data.get("retrieval_results", [])
    total_reqs = len(retrieval_results)

    # The candidate's actual skills, recovered from evidence (with hands-on flag).
    candidate_skills = _extract_candidate_skills(retrieval_results)
    cand_names = [v["display"] for v in candidate_skills.values()]

    def _relate(req: str):
        """(best_skill, cosine, relation, is_narrative) for a requirement vs the resume."""
        best, cos, rel = skill_semantics_service.find_related_skill(req, cand_names, exclude={req})
        narrative = bool(best) and candidate_skills.get(best.lower(), {}).get("narrative", False)
        return best, cos, rel, narrative

    requirements_fits = []
    missing_skills = []
    learning_roadmap = []
    covered_weight = 0.0
    total_weight = 0.0
    matched_count = 0
    exp_aligned_count = 0
    proj_aligned_count = 0
    confidence_sum = 0
    strengths = []
    weaknesses = []
    skill_relationships = []
    # Requirements the candidate can transfer into (for interview questions).
    transfer_targets = []   # (req, related_skill)
    partial_targets = []    # req listed but shallow
    matched_targets = []    # req strongly held

    for item in retrieval_results:
        req = item["requirement"]
        matches = item.get("matches", [])
        req_importance = item.get("importance", "must")
        req_weight = item.get("weight", settings.REQUIREMENT_WEIGHT_MUST)
        total_weight += req_weight
        category = skill_semantics_service.classify_category(req)

        if matches:
            # --- Direct evidence path ---
            matched_count += 1
            best_match = max(matches, key=lambda x: x.get("confidence", x.get("score", 0.0) * 100))
            best_section = best_match.get("section", "Summary")
            confidence = best_match.get("confidence")
            if confidence is None:
                sim_score = min(max(best_match.get("score", 0.0) * 100, 0), 100)
                sec_score = 100 if best_section == "Experience" else 80 if best_section == "Projects" else 30
                confidence = int(
                    (sim_score * settings.RETRIEVAL_WEIGHT_SIMILARITY) +
                    (sec_score * settings.RETRIEVAL_WEIGHT_SECTION) +
                    (30 * settings.RETRIEVAL_WEIGHT_DENSITY) +
                    (40 * settings.RETRIEVAL_WEIGHT_TECH_SPECIFICITY)
                )
                confidence = min(max(confidence, 0), 100)
            evidence_text = best_match.get("chunk", "")
            confidence_sum += confidence
            covered_weight += req_weight
            if best_section == "Experience":
                exp_aligned_count += 1
            elif best_section == "Projects":
                proj_aligned_count += 1

            # A related, hands-on skill can reinforce a shallow mention.
            rel_skill, rel_cos, relation, rel_narr = _relate(req)
            support = ""
            if relation in ("equivalent", "related") and rel_narr and rel_skill.lower() != req.lower():
                support = f" It is also reinforced by your hands-on {rel_skill} experience, which is closely related."
                skill_relationships.append(f"Your {rel_skill} experience directly supports {req}.")

            if confidence >= 75:
                status = "Matched"
                limitations = "None"
                explanation = f'Clearly demonstrated in your {best_section}: "{_short(evidence_text)}". A direct, hands-on match.' + support
                strengths.append(f"Strong, proven {req} — shown in {best_section}.")
                matched_targets.append(req)
            else:
                status = "Partial"
                explanation = f"You mention {req}, but it reads as a listed skill rather than hands-on work in a project." + support
                limitations = f"Depth of {req} is unconfirmed — no project outcome shown."
                weaknesses.append(f"{req}: mentioned but not backed by clear project work.")
                partial_targets.append(req)
                learning_roadmap.append(LearningRoadmapItem(
                    skill=req, estimated_time=PARTIAL_SKILL_LEARNING_TIME,
                    reason=f"You list {req}; a little hands-on project practice would make it interview-ready."))
        else:
            # --- No direct evidence: try equivalence rescue, then transfer note ---
            rel_skill, rel_cos, relation, rel_narr = _relate(req)

            if relation == "equivalent":
                # The candidate effectively has this skill under a sibling name.
                ev = candidate_skills.get(rel_skill.lower(), {})
                status = "Matched" if rel_narr else "Partial"
                evidence_text = ev.get("chunk", "")
                confidence = int(rel_cos * 100)
                confidence_sum += confidence
                matched_count += 1
                covered_weight += req_weight
                explanation = (f"You list {rel_skill}, which is practically the same skill as {req} — "
                               f"they are interchangeable in day-to-day work, so this is met.")
                limitations = "None" if status == "Matched" else f"Shown as a listed skill; confirm depth of {req}/{rel_skill}."
                strengths.append(f"{req} covered via your {rel_skill} experience (equivalent skills).")
                skill_relationships.append(f"{rel_skill} is equivalent to {req} — counts as met.")
                (matched_targets if status == "Matched" else partial_targets).append(req)

            elif relation == "related":
                # Adjacent skill: honest gap, but a fast ramp-up — do NOT claim it.
                status = "Missing"
                evidence_text = ""
                confidence = 0
                missing_skills.append(req)
                explanation = (f"Not shown directly. But your {rel_skill} experience is closely related, "
                               f"so picking up {req} should be quick.")
                limitations = f"No direct {req} experience yet — this is a ramp-up, not a hard gap."
                skill_relationships.append(f"{rel_skill} is closely related to {req} — expect a short ramp-up.")
                transfer_targets.append((req, rel_skill))
                (weaknesses.insert(0, f"Missing must-have: {req} (but {rel_skill} transfers).")
                 if req_importance == "must" else
                 weaknesses.append(f"Missing nice-to-have: {req} ({rel_skill} transfers)."))
                est_time, _ = _select_learning_band(rel_cos)
                learning_roadmap.append(LearningRoadmapItem(
                    skill=req, estimated_time=est_time,
                    reason=f"You already know {rel_skill} (closely related), so expect a short ramp-up on {req}."))

            else:
                # Genuine gap: no evidence and nothing related.
                status = "Missing"
                evidence_text = ""
                confidence = 0
                missing_skills.append(req)
                explanation = "No evidence of this skill, and no closely related experience in the resume."
                limitations = "A genuine gap — would need dedicated learning before the role."
                (weaknesses.insert(0, f"Missing must-have: {req} (no related experience).")
                 if req_importance == "must" else
                 weaknesses.append(f"Missing nice-to-have: {req}."))
                est_time, _ = _select_learning_band(0.0)
                learning_roadmap.append(LearningRoadmapItem(
                    skill=req, estimated_time=est_time,
                    reason=f"No related background found; {req} would need foundational learning."))

        requirements_fits.append(RequirementFit(
            requirement=req, category=category, status=status,
            matched_evidence=evidence_text, explanation=explanation, limitations=limitations,
            confidence=confidence, importance=req_importance, weight=req_weight))

    # Sub-scores. Coverage is importance-weighted over the FINAL statuses (so an
    # equivalence rescue correctly counts as covered).
    coverage_score = int((covered_weight / total_weight) * 100) if total_weight > 0 else 0
    experience_score = int((exp_aligned_count / total_reqs) * 100) if total_reqs > 0 else 0
    project_score = int((proj_aligned_count / total_reqs) * 100) if total_reqs > 0 else 0
    confidence_score = int(confidence_sum / matched_count) if matched_count > 0 else 0

    # Deterministic resume-quality + authenticity signal (keyword-stuffing / over-claim
    # detection) derived from the evidence sections — replaces the old hardcoded 85.
    authenticity_result = authenticity_service.assess_authenticity(evidence_data)
    quality_score = authenticity_result.quality_score

    # Surface a strong stuffing signal in the recruiter-facing weaknesses, prioritized.
    if (authenticity_result.assessment.keyword_stuffing_risk == "High"
            and authenticity_result.assessment.over_claimed_skills):
        over = ", ".join(authenticity_result.assessment.over_claimed_skills[:3])
        weaknesses.insert(
            0,
            f"Possible keyword stuffing: {over} listed but not demonstrated in Experience/Projects."
        )
    
    # Algorithmic weights multiplication
    overall = int(
        (coverage_score * settings.WEIGHT_COVERAGE) +
        (experience_score * settings.WEIGHT_EXPERIENCE) +
        (project_score * settings.WEIGHT_PROJECTS) +
        (confidence_score * settings.WEIGHT_CONFIDENCE) +
        (quality_score * settings.WEIGHT_QUALITY)
    )
    overall_score = min(max(overall, 0), 100)
    
    # Recruiter recommendation
    if overall_score >= 80:
        recruiter_recommendation = "Highly Recommended - Proceed to Technical Screen"
    elif overall_score >= 60:
        recruiter_recommendation = "Conditional Match - Proceed to Initial Interview"
    else:
        recruiter_recommendation = "Not Recommended - Reject"
        
    matched_n = sum(1 for f in requirements_fits if f.status == "Matched")
    summary = (
        f"Overall fit is {overall_score}%. The candidate clearly covers {matched_n} of {total_reqs} "
        f"requirements, with requirement coverage at {coverage_score}% and evidence confidence at "
        f"{confidence_score}%. "
        + (f"Some gaps are transferable from closely-related experience. "
           if transfer_targets else "")
        + ("Strong direct match overall." if overall_score >= 75
           else "A conditional match worth interviewing." if overall_score >= 60
           else "Weak fit for this role as written.")
    )

    # Interview questions that actually help the interviewer decide:
    #  1) probe transferable gaps (does the adjacent skill really carry over?)
    #  2) verify shallow/listed claims, 3) test depth on strong skills.
    questions = []
    for req, related in transfer_targets[:3]:
        questions.append(
            f"The candidate hasn't used {req} directly but has {related} experience, which is closely "
            f"related. Ask them to walk through how they would apply what they know from {related} to {req} — "
            f"their answer shows whether the transfer is real.")
    for req in partial_targets[:2]:
        questions.append(
            f"They list {req} but haven't shown depth. Ask for one specific problem they solved with {req} "
            f"and what the outcome was.")
    for req in matched_targets[:2]:
        if len(questions) >= 5:
            break
        questions.append(
            f"They are strong in {req}. Ask about a hard trade-off or a tricky bug they handled with {req} "
            f"to confirm real depth.")
    if not questions:
        questions = [
            "Walk me through a project you are most proud of and the specific role you played.",
            "Tell me about a time you had to learn a new tool quickly for a project.",
        ]
        
    # Rejection Email (Module 8)
    rejection_email = None
    if overall_score < 75:
        gap_skills = [f.requirement for f in requirements_fits if f.status in ["Partial", "Missing"]]
        gaps_list = ", ".join(gap_skills[:3]) if gap_skills else "advanced deployment tools"
        
        rejection_email = f"""Subject: Update on your application

Dear Candidate,

Thank you for sharing your resume and technical background with our recruitment team. We appreciate the time you took to compile your application.

After analyzing the semantic alignment between your projects and our core requirements, we identified gaps in: {gaps_list}. As our role demands deep technical autonomy in these specific domains, we are not able to move forward with your candidacy at this time.

We were highly impressed by your qualifications and would love to retain your details for future opportunities. We wish you the best in your career search.

Best regards,
Recruitment Team"""

    return HiringReport(
        overall_score=overall_score,
        coverage_score=coverage_score,
        experience_score=experience_score,
        project_score=project_score,
        confidence_score=confidence_score,
        quality_score=quality_score,
        summary=summary,
        requirements=requirements_fits,
        strengths=strengths[:5],
        weaknesses=weaknesses[:5],
        skill_relationships=skill_relationships[:5],
        missing_skills=missing_skills,
        learning_roadmap=learning_roadmap,
        interview_questions=questions[:6],
        recruiter_recommendation=recruiter_recommendation,
        rejection_email=rejection_email,
        authenticity=authenticity_result.assessment,
        candidate_profile=evidence_data.get("candidate_profile")
    )
