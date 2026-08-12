from datetime import datetime
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
    # Which embedding model gathers the evidence. Carried through the graph so the
    # choice is part of the run's identity rather than a global default.
    embedding_engine: Optional[str]
    # When the resume entered the platform. Feeds the Match Score's freshness
    # parameter; carried through the graph rather than read from the clock so a
    # re-run of the same analysis reproduces the same score.
    resume_uploaded_at: Optional[datetime]
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
        analysis_id=state["analysis_id"],
        embedding_engine=state.get("embedding_engine"),
        resume_uploaded_at=state.get("resume_uploaded_at"),
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
    analysis_id: int,
    embedding_engine: Optional[str] = None,
    resume_uploaded_at: Optional[datetime] = None,
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
        "embedding_engine": embedding_engine,
        "resume_uploaded_at": resume_uploaded_at,
        "evidence_report": None,
        "final_report": None
    }
    
    logger.info(f"Invoking LangGraph state workflow for Analysis ID: {analysis_id}")
    final_state = hiring_graph.invoke(initial_state)
    return final_state["final_report"] or {}
