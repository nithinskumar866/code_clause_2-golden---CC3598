import pytest
from app.workflows.hiring_workflow import run_candidate_intelligence_node, run_hiring_decision_node, StateGraph, START, END

def test_langgraph_nodes_structure():
    # 1. Verify StateGraph compilation
    from app.workflows.hiring_workflow import hiring_graph
    assert hiring_graph is not None
    
    # 2. Test Hiring Decision Node with mock state data
    mock_state = {
        "resume_id": 99,
        "resume_path": "",
        "jd_id": 99,
        "jd_path": "",
        "analysis_id": 99,
        "evidence_report": {
            "analysis_id": 99,
            "candidate_id": 99,
            "resume_id": 99,
            "jd_id": 99,
            "retrieval_results": [
                {
                    "requirement": "python",
                    "matches": [
                        {
                            "chunk": "Experienced Python backend engineer.",
                            "section": "Experience",
                            "score": 0.90,
                            "page": 1,
                            "filename": "mock.pdf",
                            "chunk_id": 1
                        }
                    ]
                }
            ]
        },
        "final_report": None
    }
    
    node_output = run_hiring_decision_node(mock_state)
    assert "final_report" in node_output
    final = node_output["final_report"]
    
    assert final["analysis_id"] == 99
    assert "overall_score" in final
    assert "coverage_score" in final
    assert "experience_score" in final
    assert len(final["requirements"]) == 1
    assert final["requirements"][0]["status"] == "Matched"
    assert final["requirements"][0]["category"] == "Programming Language"
    assert final["requirements"][0]["confidence"] >= 75
