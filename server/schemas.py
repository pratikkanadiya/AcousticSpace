from typing import List, Optional
from pydantic import BaseModel


class SegmentPrediction(BaseModel):
    segment_id: int
    spoof_probability: float


class AcousticSummary(BaseModel):
    rt60_sec: float
    drr_db: float
    room_clarity_c50: float
    room_centroid: float
    breath_count: float
    avg_breath_interval: float


class AnalysisResponse(BaseModel):
    filename: str
    duration_sec: float
    sample_rate: int
    num_segments: int
    is_clipped: bool

    verdict: str          
    confidence_pct: float   
    spoof_probability: float  

    acoustic_summary: AcousticSummary
    segment_predictions: List[SegmentPrediction]


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    acoustic_stats_loaded: bool
