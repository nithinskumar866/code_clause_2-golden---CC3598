import io

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "healthy"
    assert data["data"]["database"] == "connected"

def test_resume_upload_success(client):
    file_content = b"%PDF-1.4 dummy pdf content"
    file_data = {"file": ("test_resume.pdf", io.BytesIO(file_content), "application/pdf")}
    
    response = client.post("/api/v1/resume/upload", files=file_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["filename"] == "test_resume.pdf"
    assert data["data"]["status"] == "Uploaded"
    assert "id" in data["data"]

def test_resume_upload_invalid_extension(client):
    file_content = b"dummy text content"
    file_data = {"file": ("test_resume.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/api/v1/resume/upload", files=file_data)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UPLOAD_ERROR"

def test_job_upload_success(client):
    file_content = b"dummy docx content"
    # DOCX MIME type
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    file_data = {"file": ("test_jd.docx", io.BytesIO(file_content), docx_mime)}
    
    response = client.post("/api/v1/job/upload", files=file_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["filename"] == "test_jd.docx"
    assert data["data"]["status"] == "Uploaded"
    assert "id" in data["data"]

def test_job_upload_invalid_mime(client):
    file_content = b"dummy content"
    file_data = {"file": ("test_jd.pdf", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/api/v1/job/upload", files=file_data)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UPLOAD_ERROR"
