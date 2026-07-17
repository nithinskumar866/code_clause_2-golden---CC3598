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
        "job's requirements.\n\n"
        "For each requirement you are given its importance (must-have vs nice-to-have) "
        "and the exact evidence retrieved from the candidate's resume. Reason ONLY from "
        "this evidence — never assume a skill that is not shown. A requirement with no "
        "retrieved evidence is \"Missing\".\n\n"
        f"EVIDENCE:\n{_format_evidence_for_prompt(evidence_data)}\n\n"
        "Return a SINGLE valid JSON object (no markdown, no commentary) with EXACTLY "
        f"this shape:\n{_REASONING_SCHEMA}\n\n"
        "Guidance:\n"
        "- status: Matched (clearly demonstrated), Partial (mentioned but thin/"
        "uncorroborated), or Missing (no evidence).\n"
        "- matched_evidence: quote the specific evidence you used (empty if Missing).\n"
        "- confidence (0-100): how strongly the evidence supports the requirement.\n"
        "- Weight must-have requirements more heavily than nice-to-haves in strengths, "
        "weaknesses, and the recommendation.\n"
        "- interview_questions: target the weakest or least-verified areas.\n"
        "- rejection_email: a brief, kind draft ONLY if the candidate is clearly weak "
        "overall; otherwise null.\n"
        "- Never invent skills or evidence."
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
    Refined dynamic matching, classification, relationship reasoning, and scoring algorithm (Mock Engine).
    """
    retrieval_results = evidence_data.get("retrieval_results", [])
    total_reqs = len(retrieval_results)

    # Skills the candidate has actual resume evidence for. Used as the basis for
    # semantic transferability reasoning about missing requirements.
    demonstrated_skills = [it["requirement"] for it in retrieval_results if it.get("matches")]
    
    requirements_fits = []
    missing_skills = []
    learning_roadmap = []
    
    matched_count = 0
    exp_aligned_count = 0
    proj_aligned_count = 0
    confidence_sum = 0
    
    strengths = []
    weaknesses = []
    skill_relationships = []
    
    # 1. Evaluate every requirement structurally
    for item in retrieval_results:
        req = item["requirement"]
        matches = item.get("matches", [])
        req_importance = item.get("importance", "must")
        req_weight = item.get("weight", settings.REQUIREMENT_WEIGHT_MUST)

        # Classify the requirement into a category via semantic prototype
        # similarity. Generalizes to skills outside the known taxonomy.
        category = skill_semantics_service.classify_category(req)

        if not matches:
            status = "Missing"
            evidence_text = ""
            explanation = "No evidence found in candidate's resume."
            limitations = f"Document did not mention or imply competency in {req}."
            confidence = 0

            missing_skills.append(req)
            # A missing must-have is a far bigger red flag than a missing nice-to-have.
            if req_importance == "must":
                weaknesses.insert(0, f"Missing must-have requirement: {req} ({category}).")
            else:
                weaknesses.append(f"Missing nice-to-have: {req} ({category}).")

            # Transferability: estimate learning effort from how semantically
            # related the candidate's demonstrated skills are to this gap.
            transfer_score, related_skill, same_category = skill_semantics_service.estimate_transfer(
                req, demonstrated_skills
            )
            est_time, strength = _select_learning_band(transfer_score)

            if related_skill and strength in ("strong", "moderate"):
                relation = "closely related" if same_category else "conceptually related"
                reason = (
                    f"Candidate's experience with {related_skill} is {relation} to {req}, "
                    f"which is expected to ease the learning curve."
                )
                skill_relationships.append(
                    f"{related_skill} experience partially supports acquiring {req} "
                    f"({category}) due to semantic overlap."
                )
            else:
                reason = f"Candidate lacks background in {category} concepts; requires core instruction."

            learning_roadmap.append(
                LearningRoadmapItem(
                    skill=req,
                    estimated_time=est_time,
                    reason=reason
                )
            )
        else:
            matched_count += 1
            best_match = max(matches, key=lambda x: x.get("confidence", x.get("score", 0.0) * 100))
            best_score = best_match.get("score", 0.0)
            best_section = best_match.get("section", "Summary")
            confidence = best_match.get("confidence")
            if confidence is None:
                sim_score = min(max(best_score * 100, 0), 100)
                sec_score = 100 if best_section == "Experience" else 80 if best_section == "Projects" else 30
                density_score = 30
                tech_score = 40
                confidence = int(
                    (sim_score * settings.RETRIEVAL_WEIGHT_SIMILARITY) +
                    (sec_score * settings.RETRIEVAL_WEIGHT_SECTION) +
                    (density_score * settings.RETRIEVAL_WEIGHT_DENSITY) +
                    (tech_score * settings.RETRIEVAL_WEIGHT_TECH_SPECIFICITY)
                )
                confidence = min(max(confidence, 0), 100)
            evidence_text = best_match.get("chunk", "")
            
            if best_section == "Experience":
                exp_aligned_count += 1
            elif best_section == "Projects":
                proj_aligned_count += 1
                
            if confidence >= 75:
                status = "Matched"
                limitations = "None"
                explanation = f"Demonstrated practical applications inside {best_section} with high specificity."
                strengths.append(f"Proven competency in {req} supported by matching {best_section} references.")
            else:
                status = "Partial"
                limitations = "Listed in Skills keyword list but lacks measurable project detail."
                explanation = f"Listed under {best_section} section without project descriptions or outcomes."
                weaknesses.append(f"Competency claim for {req} is only partially supported.")
                
                # Dynamic learning pathway
                learning_roadmap.append(
                    LearningRoadmapItem(
                        skill=req,
                        estimated_time=PARTIAL_SKILL_LEARNING_TIME,
                        reason=f"Candidate lists {req} but needs hands-on project practice to establish confidence."
                    )
                )
                
            confidence_sum += confidence
            
        requirements_fits.append(
            RequirementFit(
                requirement=req,
                category=category,
                status=status,
                matched_evidence=evidence_text,
                explanation=explanation,
                limitations=limitations,
                confidence=confidence,
                importance=req_importance,
                weight=req_weight
            )
        )
        
    # Calculate sub-scores (0-100 scale). Coverage is importance-weighted so that
    # missing must-haves depress the score more than missing nice-to-haves.
    coverage_score = _weighted_coverage(retrieval_results)
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
        
    summary = (
        f"Candidate analyzed against {total_reqs} job parameters. "
        f"Algorithm calculated overall compatibility fit at {overall_score}% based on requirement coverage ({coverage_score}%), "
        f"experience section depth ({experience_score}%), and evidence confidence ({confidence_score}%)."
    )
    
    # Dynamic questions: Depth vs Verification (Module 7)
    questions = []
    for fit in requirements_fits:
        if fit.status == "Partial":
            questions.append(
                f"Verification: You list {fit.requirement} in your resume. Can you walk me through the configuration parameters or deployment settings you used in your projects?"
            )
        elif fit.status == "Matched" and len(questions) < 2:
            questions.append(
                f"Depth: You demonstrate strong experience in {fit.requirement}. Can you describe a complex troubleshooting scenario or trade-off decision you made using this tool?"
            )
            
    if not questions:
        questions = [
            "Can you describe your system design experience in microservices or monolithic structures?",
            "How do you approach learning new frameworks or deployment structures like Docker and Kubernetes?"
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
        strengths=strengths[:4],
        weaknesses=weaknesses[:4],
        skill_relationships=skill_relationships[:3],
        missing_skills=missing_skills,
        learning_roadmap=learning_roadmap,
        interview_questions=questions[:4],
        recruiter_recommendation=recruiter_recommendation,
        rejection_email=rejection_email,
        authenticity=authenticity_result.assessment,
        candidate_profile=evidence_data.get("candidate_profile")
    )
