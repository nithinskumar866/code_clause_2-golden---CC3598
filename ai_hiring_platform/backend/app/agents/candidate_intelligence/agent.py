import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from app.agents.base.interfaces import CandidateIntelligenceAgentInterface
from app.services.ai import (
    candidate_facets_service,
    document_loader,
    job_profile_extractor,
    parser_service,
    resume_structuring_service,
    embedding_service,
    embedding_store,
    vector_store_service,
    jd_parser,
    jd_requirement_extractor,
    retrieval_service,
    profile_service
)
from app.core.constants import REPORT_DIR
from app.core.logging import logger

class CandidateIntelligenceAgent(CandidateIntelligenceAgentInterface):
    """
    Candidate Intelligence Agent orchestrator.
    Coordinates document loaders, parsing, structuring, embedding, FAISS indexing,
    and semantic retrieval services without containing core AI logic.
    """
    
    def ingest_candidate_resume(
        self, resume_path: str, resume_id: int, embedding_engine: Optional[str] = None
    ) -> None:
        """
        Make one candidate searchable, in the store every part of the platform shares.

        This used to build a FAISS index PER RESUME, hardcoded to 384 dimensions — which
        meant the evaluation pipeline could only ever run on BGE, and that every resume
        was embedded a second time for the chatbot's pool index. Both now read the same
        vectors, so choosing a model is meaningful here too and nothing is embedded twice.

        Content-addressed as before: an unchanged file is not re-parsed or re-embedded.
        """
        filename = os.path.basename(resume_path)
        embedding_store.ensure_resume_searchable(
            resume_id=resume_id, path=resume_path, filename=filename, model=embedding_engine
        )
        logger.info(f"Resume ID {resume_id} is searchable on the shared store.")

    def retrieve_evidence(
        self,
        resume_id: int,
        resume_path: str,
        jd_path: str,
        jd_id: int,
        analysis_id: int,
        embedding_engine: Optional[str] = None,
        resume_uploaded_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates RAG retrieval: extracts keywords from JD, queries FAISS index
        separately per requirement, compiles evidence JSON, and saves report to disk.
        """
        logger.info(f"Orchestrating evidence retrieval: Resume ID {resume_id}, JD ID {jd_id}, Analysis ID {analysis_id}")
        
        # Self-healing: parse and index this candidate if the shared store does not
        # already hold them under this exact file, for THIS model.
        model_index = embedding_store.ensure_resume_searchable(
            resume_id=resume_id,
            path=resume_path,
            filename=os.path.basename(resume_path),
            model=embedding_engine,
        )
        if model_index.is_empty():
            raise RuntimeError(f"No vectors available for Resume ID {resume_id}.")
            
        # 2. Parse Job Description text
        jd_text = jd_parser.parse_job_description(jd_path)
        
        # 3. Extract technical requirements list
        requirements = jd_requirement_extractor.extract_requirements(jd_text)
        
        # 4. Perform requirement-wise similarity retrieval
        # Retrieve top 3 matching chunks for each requirement
        # Scored against this candidate's own chunks only, using the floor calibrated
        # for whichever model is serving — a threshold from a different model would
        # silently discard genuine evidence.
        retrieval_results = retrieval_service.retrieve_evidence_from_store(
            model_index=model_index,
            resume_id=resume_id,
            requirements=requirements,
            top_k=3,
        )

        # 4b. Attach requirement priority (must-have vs nice-to-have + weight),
        # derived from the JD wording, so downstream scoring is importance-weighted.
        priorities = jd_requirement_extractor.classify_priorities(jd_text, requirements)
        for item in retrieval_results:
            pr = priorities.get(item["requirement"], {"importance": "must", "weight": 1.0})
            item["importance"] = pr["importance"]
            item["weight"] = pr["weight"]

        # 4c. Derive a deterministic candidate profile (identity + seniority fit vs the
        # JD). Uses parsed resume text — the LLM never sees the raw document.
        candidate_profile = None
        candidate_facets = None
        resume_text = ""
        try:
            resume_text = document_loader.load_document(resume_path)
            profile = profile_service.extract_profile(resume_text, jd_text)
            candidate_profile = profile.model_dump()
            # 4d. Facets the Match Score needs beyond identity: which skills are
            # claimed versus practised, location, education, joining window. Derived
            # here because this is the evidence side — Agent 2 only reasons over what
            # this agent gathers, and must never open the document itself.
            candidate_facets = candidate_facets_service.extract_candidate_facets(
                resume_text,
                title=profile.title,
                total_years=profile.total_years,
                uploaded_at=resume_uploaded_at,
            ).to_dict()
        except Exception as e:
            logger.error(f"Candidate profile extraction skipped: {e}", exc_info=True)

        # 4e. The JD's own structured facts (title, location, industry, education bar,
        # experience range, joining window). The requirement list alone answers only
        # "which skills"; six of the nine Match Score parameters need these.
        job_profile = job_profile_extractor.extract_job_profile(jd_text, requirements).to_dict()

        # 5. Compile structured evidence report JSON
        report = {
            "analysis_id": analysis_id,
            "candidate_id": resume_id,
            "resume_id": resume_id,
            "jd_id": jd_id,
            "candidate_profile": candidate_profile,
            "candidate_facets": candidate_facets,
            "job_profile": job_profile,
            "retrieval_results": retrieval_results
        }
        
        # 6. Save report JSON to storage/reports/analysis_<id>.json
        report_path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
        logger.info(f"Saving compiled analysis evidence report to {report_path}")
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write analysis report file: {e}", exc_info=True)
            
        return report
