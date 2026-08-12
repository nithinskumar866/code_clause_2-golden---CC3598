import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup test environment variables before importing settings
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Never auto-index during tests. Creating a TestClient runs the app's startup hook, and
# a real indexing pass would load two multi-hundred-megabyte ONNX models into the test
# process and embed the operator's entire resume pool — minutes of work, gigabytes of
# memory, and results that depend on whatever happens to be on that machine. The
# indexing behaviour itself is covered by its own tests with the engines stubbed.
os.environ["AUTO_INDEX_ON_UPLOAD"] = "false"
os.environ["AUTO_INDEX_ON_STARTUP"] = "false"

from app.core.database import Base, get_db
from app.main import app

# Create test database engine
engine = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(autouse=True)
def isolate_llm_config():
    """
    Neutralise the operator's LLM configuration for every test.

    The suite asserts the *deterministic* engine's behaviour, so it must not depend on
    whether the machine running it happens to have a provider key or a self-hosted
    endpoint configured in `.env`. Tests that specifically exercise a configured LLM
    set the values they need explicitly (monkeypatch wins over this fixture, since
    this runs first).
    """
    from app.core.config import settings

    saved = {
        name: getattr(settings, name)
        for name in ("LLM_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    }
    for name in saved:
        setattr(settings, name, "")
    yield
    for name, value in saved.items():
        setattr(settings, name, value)


@pytest.fixture(scope="session", autouse=True)
def init_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

@pytest.fixture
def db_session():
    connection = engine.connect()
    session = TestingSessionLocal(bind=connection)

    try:
        yield session
    finally:
        # Guarantee per-test isolation even when the code under test commits
        # (endpoints and services do). Roll back anything pending, then wipe all
        # rows so the next test starts from a clean slate.
        session.rollback()
        session.close()
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete())
        connection.commit()
        connection.close()

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
