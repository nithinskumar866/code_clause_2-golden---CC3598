"""
API surface tests.

Deliberately shallow: the routes are thin, and everything worth asserting about crawl,
refresh and retrieval is tested against the services directly. What is checked here is
the behaviour that only exists at the HTTP boundary — the envelope shape, and what the
service does when its one datastore is missing.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_answers_even_with_no_store(self, client, monkeypatch):
        """
        The one endpoint that must never depend on Qdrant.

        A health check that fails when the database is down tells an operator nothing
        they did not already know, and hides the detail that would help.
        """
        monkeypatch.setattr(routes.qdrant, "configured", lambda: False)
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "degraded"
        assert "QDRANT_URL" in body["data"]["qdrant"]["detail"]

    def test_health_reports_the_embedding_config(self, client, monkeypatch):
        monkeypatch.setattr(routes.qdrant, "configured", lambda: False)
        data = client.get("/api/v1/health").json()["data"]
        assert data["embedding_dim"] > 0
        assert data["embedding_model"]


class TestStoreGuard:
    """Every data route refuses cleanly rather than raising a 500 from the client."""

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("get", "/api/v1/companies", None),
            ("post", "/api/v1/companies", {"domain": "acme.com"}),
            ("get", "/api/v1/companies/acme-com", None),
            ("delete", "/api/v1/companies/acme-com", None),
            ("post", "/api/v1/companies/acme-com/crawl", {}),
            ("post", "/api/v1/refresh/run", {}),
            ("post", "/api/v1/chat/query", {"message": "who is the ceo"}),
            ("post", "/api/v1/chat/stream", {"message": "who is the ceo"}),
        ],
    )
    def test_unconfigured_store_returns_503(self, client, monkeypatch, method, path, body):
        monkeypatch.setattr(routes.qdrant, "configured", lambda: False)
        response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
        assert response.status_code == 503
        assert "QDRANT_URL" in response.json()["detail"]


class TestContract:
    def test_root_points_at_the_docs(self, client):
        body = client.get("/").json()
        assert body["service"] == "company_intel"
        assert body["health"] == "/api/v1/health"

    def test_every_route_is_registered(self, client):
        paths = set(client.get("/openapi.json").json()["paths"])
        assert paths == {
            "/",
            "/api/v1/health",
            "/api/v1/companies",
            "/api/v1/companies/{company_id}",
            "/api/v1/companies/{company_id}/crawl",
            "/api/v1/refresh/run",
            "/api/v1/chat/query",
            "/api/v1/chat/stream",
        }

    def test_a_bad_domain_is_a_400_not_a_500(self, client, monkeypatch):
        monkeypatch.setattr(routes.qdrant, "configured", lambda: True)
        monkeypatch.setattr(
            routes.registry, "register", lambda **kw: (_ for _ in ()).throw(ValueError("bad domain"))
        )
        response = client.post("/api/v1/companies", json={"domain": "not a domain"})
        assert response.status_code == 400
