import os
import json
from typing import List, Dict, Any
from pydantic import ValidationError
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from app.core.config import settings
from app.core.constants import LEARNING_TRANSFER_BANDS, PARTIAL_SKILL_LEARNING_TIME
from app.core.logging import logger
from app.schemas.analysis import HiringReport, RequirementFit, LearningRoadmapItem
from app.services.ai import skill_semantics_service

def get_llm():
    """
    Instantiates and returns the configured LLM class based on settings.
    """
    provider = settings.LLM_PROVIDER.lower()
    try:
        if provider == "openai" and settings.OPENAI_API_KEY:
            return OpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
        elif provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            return Anthropic(model="claude-3-5-sonnet-20240620", api_key=settings.ANTHROPIC_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider {provider}: {e}")
    return None

def evaluate_evidence(evidence_data: Dict[str, Any]) -> HiringReport:
    """
    Algorithmic evaluation pipeline. Invokes LLM reasoning or falls back to
    a deterministic rule-based mock engine if API credentials are not found.
    """
    llm = get_llm()
    if not llm:
        logger.warning("No LLM API keys configured. Falling back to mock reasoning engine.")
        return run_mock_evaluation(evidence_data)
        
    prompt = f"""
    You are a Senior Technical Recruiter. Your task is to evaluate a candidate based on retrieved resume evidence.
    
    Candidate Evidence JSON:
    {json.dumps(evidence_data, indent=2)}
    
    You MUST analyze the evidence and return a single, valid JSON object matching this schema. Do not output anything else.
    
    JSON Schema:
    {{
      "coverage_score": 80, // percentage of requirements with matched evidence (0-100)
      "experience_score": 75, // experience alignment (0-100) based on depth in Experience sections
      "project_score": 85, // projects relevance (0-100) based on project evidence
      "confidence_score": 70, // evidence confidence (0-100) based on depth and measurability of claims
      "quality_score": 85, // general resume quality (0-100) based on layout and technical specificity
      "summary": "Executive recruitment summary...",
      "requirements": [
        {{
          "requirement": "Python",
          "category": "Programming Language", // classify dynamically: Programming Language, Framework, DevOps, etc.
          "status": "Matched", // Matched, Partial, or Missing
          "matched_evidence": "Developed REST APIs using Python...",
          "explanation": "Why this evidence matters for this requirement",
          "limitations": "What was missing or undocumented for this requirement, or 'None'",
          "confidence": 90 // confidence score (0-100) based on specificity, projects, metrics
        }}
      ],
      "strengths": [
        "Strong experience in backend API development using Python"
      ],
      "weaknesses": [
        "Lacks practical experience with container orchestration (Docker/Kubernetes)"
      ],
      "skill_relationships": [
        "Python backend experience partially supports learning fastapi curve due to syntax overlap"
      ],
      "missing_skills": ["Redis"],
      "learning_roadmap": [
        {{
          "skill": "Redis",
          "estimated_time": "3-5 days",
          "reason": "Estimate effort dynamically based on overlap with Python databases."
        }}
      ],
      "interview_questions": [
        "Verification question if confidence is low, or depth question if confidence is high..."
      ],
      "recruiter_recommendation": "Recommend for phone screen", // proced to interview, conditional screen, or reject
      "rejection_email": "Subject: ... (only generate dynamically if overall compatibility score is below 75, otherwise null)"
    }}
    
    Rules:
    - Never invent or hallucinate skills.
    - Base matches and limitations ONLY on the candidate evidence text.
    - Compute timelines and questions dynamically based on transferability relationships.
    """
    
    try:
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # Strip potential markdown code block markers
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
            
        report_data = json.loads(response_text)
        
        # Calculate overall compatibility score algorithmically using configured weights
        cov = report_data.get("coverage_score", 0)
        exp = report_data.get("experience_score", 0)
        proj = report_data.get("project_score", 0)
        conf = report_data.get("confidence_score", 0)
        qual = report_data.get("quality_score", 0)
        
        overall = int(
            (cov * settings.WEIGHT_COVERAGE) +
            (exp * settings.WEIGHT_EXPERIENCE) +
            (proj * settings.WEIGHT_PROJECTS) +
            (conf * settings.WEIGHT_CONFIDENCE) +
            (qual * settings.WEIGHT_QUALITY)
        )
        
        report_data["overall_score"] = min(max(overall, 0), 100)
        
        return HiringReport.model_validate(report_data)
        
    except Exception as e:
        logger.error(f"LLM Structured evaluation failed: {e}. Falling back to mock engine.", exc_info=True)
        return run_mock_evaluation(evidence_data)

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
            weaknesses.append(f"Missing core requirements for {req} ({category}).")

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
                confidence=confidence
            )
        )
        
    # Calculate sub-scores (0-100 scale)
    coverage_score = int((matched_count / total_reqs) * 100) if total_reqs > 0 else 0
    experience_score = int((exp_aligned_count / total_reqs) * 100) if total_reqs > 0 else 0
    project_score = int((proj_aligned_count / total_reqs) * 100) if total_reqs > 0 else 0
    confidence_score = int(confidence_sum / matched_count) if matched_count > 0 else 0
    quality_score = 85
    
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
        rejection_email=rejection_email
    )
