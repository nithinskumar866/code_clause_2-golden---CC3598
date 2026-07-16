from app.services.ai.document_loader import load_document
from app.core.logging import logger

def parse_job_description(file_path: str) -> str:
    """
    Extracts raw text from Job Description using the central document loading pipeline.
    """
    logger.info(f"Parsing Job Description file: {file_path}")
    text = load_document(file_path)
    logger.info(f"Job Description parsing complete. Character length: {len(text)}")
    return text
