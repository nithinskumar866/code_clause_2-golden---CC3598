from datetime import datetime
from sqlalchemy import Column, Float, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default="Uploaded", nullable=False)  # Status: Uploaded, Indexed, Analysed, Failed
    # SHA-256 of the file bytes. Identifies the same CV re-uploaded under a different
    # filename, which would otherwise be indexed twice and appear twice in every
    # search result. Indexed for a fast lookup on upload. Nullable for rows created
    # before this column existed.
    content_hash = Column(String, nullable=True, index=True)

    # Relationship to analyses
    analyses = relationship("Analysis", back_populates="resume", cascade="all, delete-orphan")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default="Uploaded", nullable=False)  # Status: Uploaded, Indexed, Analysed, Failed

    # Relationship to analyses
    analyses = relationship("Analysis", back_populates="jd", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    jd_id = Column(Integer, ForeignKey("job_descriptions.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, default="Uploaded", nullable=False)  # Pipeline: Uploaded, Indexed, Analysed, Failed
    # Candidate hiring workflow stage (distinct from the pipeline status above).
    workflow_status = Column(String, default="Applied", server_default="Applied", nullable=False)
    # Denormalized scores from the hiring report, backfilled on read. Enable
    # efficient SQL aggregation/filtering without loading the per-analysis report
    # files. NULL until the analysis has a completed report.
    # The Match Score, carried to one decimal (78.4) because that is the agreed
    # reporting precision — rounding to a whole number here would hide real
    # differences between candidates and between implementations of the spec.
    # SQLite stores a non-integral value in an INTEGER-affinity column as REAL, so
    # databases created before this widening keep working without a rewrite.
    overall_score = Column(Float, nullable=True, index=True)
    coverage_score = Column(Integer, nullable=True)
    experience_score = Column(Integer, nullable=True)
    project_score = Column(Integer, nullable=True)
    quality_score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    resume = relationship("Resume", back_populates="analyses")
    jd = relationship("JobDescription", back_populates="analyses")
    notes = relationship(
        "RecruiterNote",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )


class RecruiterNote(Base):
    __tablename__ = "recruiter_notes"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text = Column(Text, nullable=False)
    author = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship back to the owning analysis
    analysis = relationship("Analysis", back_populates="notes")


class PromptSuite(Base):
    """
    A saved prompt plus the cases it is graded on — the regression unit of the Prompt Lab.

    The cases live beside the prompt because they only mean anything together: a case
    asserts what THIS prompt promised (1-2 lines, link only when relevant), and a suite
    detached from its prompt grades nothing.
    """

    __tablename__ = "prompt_suites"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=False, default="")
    prompt = Column(Text, nullable=False)
    # JSON array of cases. Free-form by design: a case's expectations grow as the prompt
    # grows new clauses, and a migration per clause would make that unaffordable.
    cases = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    runs = relationship("PromptRun", back_populates="suite", cascade="all, delete-orphan")


class PromptRun(Base):
    """
    One graded execution of one prompt variant over a suite.

    `prompt_hash` and `model` are stored with the score because a pass rate without them
    is not a result: the same suite scores differently against a different model, and a
    regression history that cannot tell a prompt edit from a model swap is noise.
    """

    __tablename__ = "prompt_runs"

    id = Column(Integer, primary_key=True, index=True)
    suite_id = Column(
        Integer, ForeignKey("prompt_suites.id", ondelete="CASCADE"), nullable=True, index=True
    )
    label = Column(String, nullable=False, default="A")
    prompt_hash = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False, default="")
    total_cases = Column(Integer, nullable=False, default=0)
    clean_cases = Column(Integer, nullable=False, default=0)
    passed = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    score = Column(Float, nullable=True)
    # Full per-case, per-rule detail as JSON, so a past run can be reopened rather than
    # merely remembered as a number.
    results = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    suite = relationship("PromptSuite", back_populates="runs")
