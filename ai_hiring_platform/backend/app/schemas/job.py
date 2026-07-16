from datetime import datetime
from pydantic import BaseModel

class JobDescriptionBase(BaseModel):
    filename: str

class JobDescriptionCreate(JobDescriptionBase):
    pass

class JobDescriptionResponse(JobDescriptionBase):
    id: int
    upload_time: datetime
    status: str

    class Config:
        from_attributes = True
