from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings
from app.core.logging import logger

# SQLAlchemy Engine
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class
Base = declarative_base()

def get_db():
    """
    Database session dependency.
    Yields a session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Columns added to `analyses` after the table's original creation, with the DDL
# to add each. SQLite's create_all never ALTERs existing tables, so these are
# applied idempotently by ensure_schema().
_ANALYSES_ADDED_COLUMNS = {
    "workflow_status": "ALTER TABLE analyses ADD COLUMN workflow_status VARCHAR NOT NULL DEFAULT 'Applied'",
    "overall_score": "ALTER TABLE analyses ADD COLUMN overall_score INTEGER",
}


def ensure_schema(bind=None):
    """
    Additive, idempotent backfill for columns introduced after a database was
    first created. SQLite's create_all creates missing tables but never ALTERs
    existing ones, so newly-added model columns must be added here. Safe to run
    repeatedly and never drops or rewrites existing data.
    """
    bind = bind or engine
    if not str(bind.url).startswith("sqlite"):
        return
    with bind.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(analyses)"))}
        # Empty means the table doesn't exist yet; create_all will build it with
        # the full, current column set, so there is nothing to backfill.
        if not existing:
            return
        added = False
        for column, ddl in _ANALYSES_ADDED_COLUMNS.items():
            if column not in existing:
                logger.info(f"Backfilling column analyses.{column} ...")
                conn.execute(text(ddl))
                added = True
        if added:
            conn.commit()


def init_db():
    """
    Creates all tables in the database, then applies additive schema backfills.
    """
    try:
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=engine)
        ensure_schema()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}", exc_info=True)
        raise e
