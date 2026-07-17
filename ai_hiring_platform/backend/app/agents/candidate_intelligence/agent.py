import os
import json
from typing import Dict, Any
from app.agents.base.interfaces import CandidateIntelligenceAgentInterface
from app.services.ai import (
    document_loader,
    parser_service,
    resume_structuring_service,
    embedding_service,
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
    
    def ingest_candidate_resume(self, resume_path: str, resume_id: int) -> None:
        """
        Orchestrates resume ingestion: reads file, sections it semantically,
        generates local embeddings, and persists FAISS index to disk.
        """
        if vector_store_service.has_existing_index(resume_id):
            logger.info(f"Resume ID {resume_id} already has a persisted vector index. Skipping ingestion.")
            return

        logger.info(f"Starting ingestion process for Resume ID: {resume_id}, path: {resume_path}")
        
        # 1. Load document text
        raw_text = document_loader.load_document(resume_path)
        
        # 2. Section text into LlamaIndex TextNodes
        filename = os.path.basename(resume_path)
        # Use resume_id as candidate_id in Sprint 2 simple model context
        nodes = resume_structuring_service.structure_resume_to_nodes(
            text=raw_text,
            candidate_id=resume_id,
            resume_id=resume_id,
            filename=filename
        )
        
        # 3. Generate embeddings for Nodes
        nodes_embedded = embedding_service.generate_embeddings_for_nodes(nodes)
        
        # 4. Save to FAISS vector index
        vector_store_service.save_nodes_to_index(nodes_embedded, resume_id)
        logger.info(f"Ingestion process finished for Resume ID: {resume_id}")

    def retrieve_evidence(
        self, 
        resume_id: int, 
        resume_path: str,
        jd_path: str, 
        jd_id: int,
        analysis_id: int
    ) -> Dict[str, Any]:
        """
        Orchestrates RAG retrieval: extracts keywords from JD, queries FAISS index
        separately per requirement, compiles evidence JSON, and saves report to disk.
        """
        logger.info(f"Orchestrating evidence retrieval: Resume ID {resume_id}, JD ID {jd_id}, Analysis ID {analysis_id}")
        
        # Self-healing index load
        if not vector_store_service.has_existing_index(resume_id):
            logger.info(f"Index not found for Resume ID {resume_id}. Executing ingestion pipeline first...")
            self.ingest_candidate_resume(resume_path, resume_id)
            
        # 1. Load FAISS index
        index = vector_store_service.load_index(resume_id)
        if not index:
            raise RuntimeError(f"Failed to initialize or load vector index for Resume ID: {resume_id}")
            
        # 2. Parse Job Description text
        jd_text = jd_parser.parse_job_description(jd_path)
        
        # 3. Extract technical requirements list
        requirements = jd_requirement_extractor.extract_requirements(jd_text)
        
        # 4. Perform requirement-wise similarity retrieval
        # Retrieve top 3 matching chunks for each requirement
        retrieval_results = retrieval_service.retrieve_evidence_for_requirements(
            index=index,
            requirements=requirements,
            top_k=3
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
        try:
            resume_text = document_loader.load_document(resume_path)
            candidate_profile = profile_service.extract_profile(resume_text, jd_text).model_dump()
        except Exception as e:
            logger.error(f"Candidate profile extraction skipped: {e}", exc_info=True)

        # 5. Compile structured evidence report JSON
        report = {
            "analysis_id": analysis_id,
            "candidate_id": resume_id,
            "resume_id": resume_id,
            "jd_id": jd_id,
            "candidate_profile": candidate_profile,
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
