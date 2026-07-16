---
name: langgraph
description: LangGraph workflow conventions for the AI Hiring Platform — the sequential two-node state graph, AgentState shape, and how to add nodes safely. Use when touching workflows/hiring_workflow.py or the agent orchestration.
---

# LangGraph Workflow

## The graph (keep it simple)
`workflows/hiring_workflow.py`:
```
START → candidate_intelligence → hiring_decision → END
```
Sequential. **No branching, no loops, no extra runtime agents.** Simplicity is a feature: every node is inspectable and testable, every state transition explainable.

## State
`AgentState` (TypedDict) carries: `resume_id, resume_path, jd_id, jd_path, analysis_id, evidence_report, final_report`.
- `candidate_intelligence` node fills `evidence_report` (structured evidence).
- `hiring_decision` node consumes `evidence_report` and fills `final_report`.

## Node rules
- A node calls the corresponding agent's method and returns a partial state dict.
- The decision node must read only `evidence_report` — never re-open files.
- Keep nodes thin: orchestration only; logic lives in services.

## Adding a step (rare)
Only if a genuinely new stage is needed, and without becoming a new *runtime agent*:
- Add a node function, wire it with `add_edge` preserving linear order, extend `AgentState`.
- Add/extend `tests/test_workflows.py` to assert the new state transition.
- Prefer extending an existing agent/service over adding a node.

## Verify
`tests/test_workflows.py` checks graph compilation and node outputs. Keep it green.
