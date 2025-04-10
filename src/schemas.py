from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class WorkerRecommendation(BaseModel):
    workerId: int
    name: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[WorkerRecommendation]

class TrainingResponse(BaseModel):
    message: str
    status: str

class CacheStatusResponse(BaseModel):
    last_updated: Optional[datetime]
    genres_cached: int
    is_fresh: bool