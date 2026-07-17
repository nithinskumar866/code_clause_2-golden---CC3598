import pytest

from app.models.database import Resume, JobDescription, Analysis


def _seed_analysis(db):
    """Create an Analysis (with its Resume/JD) to attach notes to."""
    resume = Resume(filename="r.pdf", status="Indexed")
    jd = JobDescription(filename="j.docx", status="Indexed")
    db.add_all([resume, jd])
    db.commit()
    db.refresh(resume)
    db.refresh(jd)

    analysis = Analysis(resume_id=resume.id, jd_id=jd.id, status="Analysed")
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis.id


def _create_note(client, analysis_id, text="Great candidate", author="Alice"):
    return client.post(
        f"/api/v1/analysis/{analysis_id}/notes",
        json={"text": text, "author": author},
    )


# ------------------------------- Create -------------------------------

def test_create_note(client, db_session):
    analysis_id = _seed_analysis(db_session)
    response = _create_note(client, analysis_id, text="Strong Python skills", author="Bob")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    note = body["data"]
    assert note["analysis_id"] == analysis_id
    assert note["text"] == "Strong Python skills"
    assert note["author"] == "Bob"
    assert note["created_at"] is not None
    assert note["updated_at"] is not None


def test_create_note_trims_whitespace(client, db_session):
    analysis_id = _seed_analysis(db_session)
    response = _create_note(client, analysis_id, text="  padded note  ", author="  Eve  ")
    assert response.status_code == 200
    assert response.json()["data"]["text"] == "padded note"
    assert response.json()["data"]["author"] == "Eve"


def test_create_note_validation_rejects_blank(client, db_session):
    analysis_id = _seed_analysis(db_session)
    assert _create_note(client, analysis_id, text="", author="Bob").status_code == 422
    assert _create_note(client, analysis_id, text="   ", author="Bob").status_code == 422
    assert _create_note(client, analysis_id, text="ok", author="").status_code == 422


def test_create_note_analysis_not_found(client, db_session):
    response = _create_note(client, 999999)
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NOT_FOUND"


# -------------------------------- Read --------------------------------

def test_list_notes_newest_first(client, db_session):
    analysis_id = _seed_analysis(db_session)
    id1 = _create_note(client, analysis_id, text="first").json()["data"]["id"]
    id2 = _create_note(client, analysis_id, text="second").json()["data"]["id"]

    response = client.get(f"/api/v1/analysis/{analysis_id}/notes")
    assert response.status_code == 200
    data = response.json()["data"]
    assert [n["id"] for n in data] == [id2, id1]  # newest first


def test_list_notes_analysis_not_found(client, db_session):
    response = client.get("/api/v1/analysis/999999/notes")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ------------------------------- Update -------------------------------

def test_update_note(client, db_session):
    analysis_id = _seed_analysis(db_session)
    note_id = _create_note(client, analysis_id, text="old", author="Alice").json()["data"]["id"]

    response = client.put(f"/api/v1/notes/{note_id}", json={"text": "updated text"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["text"] == "updated text"
    assert data["author"] == "Alice"  # unchanged


def test_update_note_requires_a_field(client, db_session):
    analysis_id = _seed_analysis(db_session)
    note_id = _create_note(client, analysis_id).json()["data"]["id"]
    # Empty body -> model validator rejects
    assert client.put(f"/api/v1/notes/{note_id}", json={}).status_code == 422
    # Blank text -> field validator rejects
    assert client.put(f"/api/v1/notes/{note_id}", json={"text": "  "}).status_code == 422


def test_update_note_not_found(client, db_session):
    response = client.put("/api/v1/notes/999999", json={"text": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ------------------------------- Delete -------------------------------

def test_delete_note(client, db_session):
    analysis_id = _seed_analysis(db_session)
    note_id = _create_note(client, analysis_id).json()["data"]["id"]

    response = client.delete(f"/api/v1/notes/{note_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Now gone
    assert client.get(f"/api/v1/analysis/{analysis_id}/notes").json()["data"] == []
    assert client.delete(f"/api/v1/notes/{note_id}").status_code == 404
