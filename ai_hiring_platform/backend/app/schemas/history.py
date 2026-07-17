from enum import Enum

from pydantic import BaseModel, Field


class RecommendationFilter(str, Enum):
    """Recruiter recommendation buckets, derived from overall_score thresholds."""
    selected = "Selected"
    borderline = "Borderline"
    rejected = "Rejected"


class HistorySort(str, Enum):
    """Supported orderings for the history search."""
    newest = "newest"
    oldest = "oldest"
    highest_score = "highest_score"
    lowest_score = "lowest_score"


class HistoryPageMeta(BaseModel):
    """Pagination metadata returned in the response envelope's `meta` field."""
    total_count: int = Field(..., description="Total analyses matching the filters")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Items per page (equals total_count when unpaged)")
    total_pages: int = Field(..., description="Number of pages for the current page_size")
