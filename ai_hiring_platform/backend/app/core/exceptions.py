from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class AppException(Exception):
    def __init__(self, message: str, code: str = "INTERNAL_SERVER_ERROR", status_code: int = 500, details: str = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or message

class UploadError(AppException):
    def __init__(self, message: str, details: str = None):
        super().__init__(message, code="UPLOAD_ERROR", status_code=400, details=details)

class DatabaseError(AppException):
    def __init__(self, message: str, details: str = None):
        super().__init__(message, code="DATABASE_ERROR", status_code=500, details=details)

class NotFoundError(AppException):
    def __init__(self, message: str, details: str = None):
        super().__init__(message, code="NOT_FOUND", status_code=404, details=details)

class CorruptedReportError(AppException):
    def __init__(self, message: str, details: str = None):
        super().__init__(message, code="CORRUPTED_REPORT", status_code=422, details=details)

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.message,
                "error": {
                    "code": exc.code,
                    "details": exc.details
                }
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected error occurred.",
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "details": str(exc)
                }
            }
        )
