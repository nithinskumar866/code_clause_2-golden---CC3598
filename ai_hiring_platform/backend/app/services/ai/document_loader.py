import os
from app.core.constants import ALLOWED_EXTENSIONS
from app.core.exceptions import UploadError
from app.services.ai.parser_service import parse_pdf, parse_docx
from app.core.logging import logger

def load_document(file_path: str) -> str:
    """
    Inspects the file extension and delegates parsing to the appropriate parser.
    """
    if not os.path.exists(file_path):
        logger.error(f"Document file not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path.lower())
    
    if ext not in ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported file extension: {ext}")
        raise UploadError(f"Unsupported file extension: {ext}. Only PDF and DOCX are allowed.")
        
    if ext == ".pdf":
        logger.info(f"Routing {file_path} to PDF parser.")
        return parse_pdf(file_path)
    elif ext == ".docx":
        logger.info(f"Routing {file_path} to DOCX parser.")
        return parse_docx(file_path)
    else:
        # Fallback (should be blocked by validation)
        raise UploadError(f"Unsupported file format: {ext}")
