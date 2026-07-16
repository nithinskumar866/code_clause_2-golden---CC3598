import os
import json
from typing import Dict, Any
from app.agents.base.interfaces import HiringDecisionAgentInterface
from app.services.ai import evaluation_service
from app.core.constants import REPORT_DIR
from app.core.logging import logger

class HiringDecisionAgent(HiringDecisionAgentInterface):
    """
    Hiring Decision Agent orchestrator.
    Consumes evidence output from CandidateIntelligenceAgent and runs evaluation logic.
    """
    
    def evaluate_candidate(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the LLM evaluation pipeline over structured RAG evidence.
        Saves and returns the enriched report JSON.
        """
        analysis_id = evidence_data.get("analysis_id")
        logger.info(f"HiringDecisionAgent initiating evaluation for Analysis ID: {analysis_id}")
        
        # 1. Run evaluation reasoning (either live LLM or mock fallback)
        report = evaluation_service.evaluate_evidence(evidence_data)
        
        # Convert Pydantic report model to dictionary
        report_dict = report.model_dump()
        
        # 2. Compile unified report: enrich evidence data with evaluation outputs
        unified_report = {
            **evidence_data,
            **report_dict
        }
        
        # 3. Save report JSON back to storage/reports/analysis_<id>.json
        if analysis_id:
            report_path = os.path.join(REPORT_DIR, f"analysis_{analysis_id}.json")
            logger.info(f"Saving enriched final recruiter report to {report_path}")
            try:
                with open(report_path, "w") as f:
                    json.dump(unified_report, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save final report file: {e}", exc_info=True)
                
        return unified_report
