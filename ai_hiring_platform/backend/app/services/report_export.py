import os
import json
import textwrap
from typing import Tuple

import fitz  # PyMuPDF (already a project dependency)
from pydantic import ValidationError

from app.services import analysis as history_service
from app.schemas.analysis import HiringReport
from app.core.exceptions import NotFoundError, CorruptedReportError
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Report Export Service.
#
# Reads an already-generated report from disk (storage/reports/analysis_<id>.json)
# and renders it as JSON or PDF. Reports are NEVER regenerated here; the existing
# HiringReport model is reused only to validate integrity. All export/serialization
# logic is centralized in this module.
# ---------------------------------------------------------------------------

_A4 = (595, 842)
_MARGIN = 50
_WRAP = 95


def _load_stored_report(analysis_id: int) -> dict:
    """Load and integrity-check a stored report. Raises NotFoundError (missing)
    or CorruptedReportError (unreadable / malformed)."""
    path = history_service._report_path(analysis_id)
    if not os.path.exists(path):
        raise NotFoundError(f"No report found for analysis {analysis_id}.")

    try:
        with open(path, "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.error(f"Report file for analysis {analysis_id} is unreadable: {e}")
        raise CorruptedReportError(f"Report for analysis {analysis_id} could not be read.", details=str(e))

    # Reuse the existing report model to detect a corrupted/incomplete report.
    try:
        HiringReport.model_validate(report)
    except ValidationError as e:
        logger.error(f"Report file for analysis {analysis_id} failed validation.")
        raise CorruptedReportError(f"Report for analysis {analysis_id} is malformed.", details=str(e))

    return report


def export_json(analysis_id: int) -> bytes:
    """Return the stored report as pretty-printed JSON bytes (no regeneration)."""
    report = _load_stored_report(analysis_id)
    return json.dumps(report, indent=2).encode("utf-8")


def export_pdf(analysis_id: int) -> bytes:
    """Return the stored report rendered as a PDF (no regeneration)."""
    report = _load_stored_report(analysis_id)
    return _render_pdf(report)


def export_filename(analysis_id: int, fmt: str) -> str:
    return f"report_analysis_{analysis_id}.{fmt}"


_EXPORTERS = {
    "json": ("application/json", export_json),
    "pdf": ("application/pdf", export_pdf),
}


def export_report(analysis_id: int, fmt: str) -> Tuple[bytes, str, str]:
    """Central entry point: (content_bytes, media_type, download_filename)."""
    media_type, exporter = _EXPORTERS[fmt]
    content = exporter(analysis_id)
    return content, media_type, export_filename(analysis_id, fmt)


# --------------------------------- PDF rendering ---------------------------------

def _render_pdf(report: dict) -> bytes:
    """Lay out the report onto A4 pages using PyMuPDF, paginating as needed."""
    doc = fitz.open()
    width, height = _A4
    state = {"page": None, "y": 0.0}

    def _new_page():
        state["page"] = doc.new_page(width=width, height=height)
        state["y"] = _MARGIN

    def _write(text: str = "", size: int = 11, gap: int = 4, bold: bool = False):
        fontname = "hebo" if bold else "helv"
        for line in (textwrap.wrap(text, _WRAP) or [""]):
            if state["y"] > height - _MARGIN:
                _new_page()
            state["page"].insert_text((_MARGIN, state["y"]), line, fontsize=size, fontname=fontname)
            state["y"] += size + gap

    _new_page()
    _write("Hiring Analysis Report", size=18, gap=12, bold=True)
    _write(f"Analysis ID: {report.get('analysis_id', '-')}", size=10, gap=10)

    _write("Scores", size=14, gap=6, bold=True)
    _write(
        f"Overall: {report.get('overall_score', 0)}    "
        f"Coverage: {report.get('coverage_score', 0)}    "
        f"Experience: {report.get('experience_score', 0)}    "
        f"Project: {report.get('project_score', 0)}    "
        f"Quality: {report.get('quality_score', 0)}"
    )
    _write(f"Recommendation: {report.get('recruiter_recommendation', '-')}", gap=10)

    _write("Summary", size=14, gap=6, bold=True)
    _write(report.get("summary", ""), gap=10)

    requirements = report.get("requirements", [])
    if requirements:
        _write("Requirements", size=14, gap=6, bold=True)
        for req in requirements:
            _write(f"- {req.get('requirement', '')}: {req.get('status', '')} "
                   f"(confidence {req.get('confidence', 0)})")
        state["y"] += 6

    for title, key in [
        ("Strengths", "strengths"),
        ("Weaknesses", "weaknesses"),
        ("Missing Skills", "missing_skills"),
        ("Interview Questions", "interview_questions"),
    ]:
        items = report.get(key, [])
        if items:
            _write(title, size=14, gap=6, bold=True)
            for item in items:
                _write(f"- {item}")
            state["y"] += 6

    roadmap = report.get("learning_roadmap", [])
    if roadmap:
        _write("Learning Roadmap", size=14, gap=6, bold=True)
        for step in roadmap:
            _write(f"- {step.get('skill', '')}: {step.get('estimated_time', '')} - {step.get('reason', '')}")

    return doc.tobytes()
