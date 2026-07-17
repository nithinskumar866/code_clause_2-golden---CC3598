from typing import Any, Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: Optional[T] = None
    # Optional response metadata (e.g. pagination). Additive and backward
    # compatible: endpoints that don't set it serialize `meta: null`.
    meta: Optional[Any] = None
