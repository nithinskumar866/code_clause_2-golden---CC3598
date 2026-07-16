from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from app.agents.candidate_intelligence.agent import CandidateIntelligenceAgent
from app.agents.hiring_decision.agent import HiringDecisionAgent
from app.core.logging import logger

class AgentState(TypedDict):
    resume_id: int
    resume_path: str
    jd_id: int
    jd_path: str
    analysis_id: int
    # Structured evidence output from Candidate Intelligence Agent
    evidence_report: Optional[Dict[str, Any]]
    # Final enriched compatibility report output from Hiring Decision Agent
    final_report: Optional[Dict[str, Any]]

candidate_agent = CandidateIntelligenceAgent()
decision_agent = HiringDecisionAgent()

def run_candidate_intelligence_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Coordinates document parsing, semantic chunking, and FAISS RAG evidence retrieval.
    """
    logger.info(f"LangGraph execution: running Candidate Intelligence Agent Node. Analysis ID: {state['analysis_id']}")
    evidence_report = candidate_agent.retrieve_evidence(
        resume_id=state["resume_id"],
        resume_path=state["resume_path"],
        jd_id=state["jd_id"],
        jd_path=state["jd_path"],
        analysis_id=state["analysis_id"]
    )
    return {"evidence_report": evidence_report}

def run_hiring_decision_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Consumes structured evidence from state, performs LLM reasoning and score normalization.
    """
    logger.info(f"LangGraph execution: running Hiring Decision Agent Node. Analysis ID: {state['analysis_id']}")
    evidence = state["evidence_report"] or {}
    final_report = decision_agent.evaluate_candidate(evidence)
    return {"final_report": final_report}

# Initialize StateGraph
workflow = StateGraph(AgentState)

# Add two agent nodes
workflow.add_node("candidate_intelligence", run_candidate_intelligence_node)
workflow.add_node("hiring_decision", run_hiring_decision_node)

# Set sequential edges
workflow.add_edge(START, "candidate_intelligence")
workflow.add_edge("candidate_intelligence", "hiring_decision")
workflow.add_edge("hiring_decision", END)

# Compile graph workflow
hiring_graph = workflow.compile()

def execute_hiring_pipeline(
    resume_id: int,
    resume_path: str,
    jd_id: int,
    jd_path: str,
    analysis_id: int
) -> Dict[str, Any]:
    """
    Orchestrates Candidate Intelligence Agent and Hiring Decision Agent sequentially using LangGraph.
    Returns the compiled final explainable report.
    """
    initial_state = {
        "resume_id": resume_id,
        "resume_path": resume_path,
        "jd_id": jd_id,
        "jd_path": jd_path,
        "analysis_id": analysis_id,
        "evidence_report": None,
        "final_report": None
    }
    
    logger.info(f"Invoking LangGraph state workflow for Analysis ID: {analysis_id}")
    final_state = hiring_graph.invoke(initial_state)
    return final_state["final_report"] or {}
