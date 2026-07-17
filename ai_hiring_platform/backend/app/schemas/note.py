from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _stripped_non_blank(value: Optional[str]) -> Optional[str]:
    """Trim surrounding whitespace and reject empty/whitespace-only strings."""
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must not be empty or whitespace")
    return trimmed


class NoteCreate(BaseModel):
    """Payload for creating a recruiter note."""
    text: str = Field(..., min_length=1, max_length=5000, description="Recruiter note body")
    author: str = Field(..., min_length=1, max_length=255, description="Author of the note")

    @field_validator("text", "author")
    @classmethod
    def _validate(cls, v: str) -> str:
        return _stripped_non_blank(v)


class NoteUpdate(BaseModel):
    """Payload for updating a recruiter note (partial: at least one field)."""
    text: Optional[str] = Field(None, min_length=1, max_length=5000)
    author: Optional[str] = Field(None, min_length=1, max_length=255)

    @field_validator("text", "author")
    @classmethod
    def _validate(cls, v: Optional[str]) -> Optional[str]:
        return _stripped_non_blank(v)

    @model_validator(mode="after")
    def _require_one(self):
        if self.text is None and self.author is None:
            raise ValueError("At least one of 'text' or 'author' must be provided.")
        return self


class NoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    analysis_id: int
    text: str
    author: str
    created_at: datetime
    updated_at: datetime
