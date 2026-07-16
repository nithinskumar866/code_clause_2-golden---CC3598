from datetime import datetime
from pydantic import BaseModel

class ResumeBase(BaseModel):
    filename: str

class ResumeCreate(ResumeBase):
    pass

class ResumeResponse(ResumeBase):
    id: int
    upload_time: datetime
    status: str

    class Config:
        from_attributes = True
