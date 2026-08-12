"""
Streaming progress tests.

The loader used to be a timer guessing at durations. These pin that the backend emits
its REAL stages, in order, and always finishes with the same payload `/query` returns —
so the UI can never drift out of step with what the pipeline is actually doing.
"""
import json


def _parse_sse(body: str):
    """[(event, payload)] from a Server-Sent Events response body."""
    out = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        event, data = None, None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if event:
            out.append((event, data))
    return out


def test_stream_emits_real_stages_then_a_result(client):
    response = client.post("/api/v1/chat/stream", json={
        "message": "who has java experience", "session_id": "stream-1",
        "limit": 2, "use_llm": False,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    kinds = [e for e, _ in events]
    assert "result" in kinds, f"stream never produced a result: {kinds}"
    assert kinds[-1] == "result", "the result must be the final event"
    assert "stage" in kinds, "no progress was reported"

    stages = [p["stage"] for e, p in events if e == "stage"]
    # Guardrails must run before anything is spent, so they lead every sequence.
    assert stages[0] == "loading"
    assert "guardrails" in stages
    # With an empty test pool the pipeline short-circuits after the guardrail check,
    # so the later stages only appear when there is a corpus to search.
    if "understanding" in stages:
        assert stages.index("guardrails") < stages.index("understanding")
    if "retrieving" in stages:
        assert stages.index("understanding") < stages.index("retrieving")
    for payload in (p for e, p in events if e == "stage"):
        assert payload["detail"], "every stage must carry human-readable text"


def test_stream_result_matches_the_plain_endpoint(client):
    payload = {"message": "who has java experience", "session_id": "stream-2",
               "limit": 2, "use_llm": False}
    plain = client.post("/api/v1/chat/query", json={**payload, "session_id": "plain-2"})
    streamed = client.post("/api/v1/chat/stream", json=payload)

    result = next(p for e, p in _parse_sse(streamed.text) if e == "result")
    plain_data = plain.json()["data"]
    assert set(result) == set(plain_data), "streamed payload must share the /query contract"
    assert result["refused"] == plain_data["refused"]


def test_refused_question_still_streams_a_result(client):
    """A guardrail refusal must close the stream cleanly, not hang the UI."""
    response = client.post("/api/v1/chat/stream", json={
        "message": "what is the capital of France", "session_id": "stream-3",
        "limit": 2, "use_llm": False,
    })
    result = next(p for e, p in _parse_sse(response.text) if e == "result")
    assert result["refused"] is True
