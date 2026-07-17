from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default="Uploaded", nullable=False)  # Status: Uploaded, Indexed, Analysed, Failed

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
