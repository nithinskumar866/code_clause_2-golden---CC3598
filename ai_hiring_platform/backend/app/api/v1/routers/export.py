import hashlib

from fastapi import APIRouter, Response

from app.core.logging import logger
from app.services import report_export

router = APIRouter()


def _download(analysis_id: int, fmt: str) -> Response:
    """Centralized download response: correct Content-Type, filename and cache headers."""
    content, media_type, filename = report_export.export_report(analysis_id, fmt)
    etag = hashlib.md5(content).hexdigest()
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "private, max-age=3600",
        "ETag": f'"{etag}"',
    }
    return Response(content=content, media_type=media_type, headers=headers)


@router.get("/analysis/{analysis_id}/export/json")
def export_report_json(analysis_id: int):
    """Download a stored hiring report as JSON."""
    logger.info(f"Export JSON requested for analysis {analysis_id}.")
    return _download(analysis_id, "json")


@router.get("/analysis/{analysis_id}/export/pdf")
def export_report_pdf(analysis_id: int):
    """Download a stored hiring report as PDF."""
    logger.info(f"Export PDF requested for analysis {analysis_id}.")
    return _download(analysis_id, "pdf")
