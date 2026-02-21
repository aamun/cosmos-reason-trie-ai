from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone

ActorType = Literal["car","truck","motorcycle","bus","bicycle","pedestrian","unknown"]
RiskLevel = Literal["low","medium","high"]

class Actor(BaseModel):
    id: str
    type: ActorType
    notes: str | None = None

class TimelineEvent(BaseModel):
    t_start: float
    t_end: float
    event: str
    evidence: list[str] = Field(default_factory=list)

class RiskAssessment(BaseModel):
    level: RiskLevel
    why: str

class CausalLink(BaseModel):
    cause: str
    effect: str

class ReportMetadata(BaseModel):
    model: str
    frame_sampling: dict
    generated_at: str

class TrafficIncidentEvidenceReport(BaseModel):
    video_id: str
    summary: str
    actors: list[Actor]
    timeline: list[TimelineEvent]
    risk_assessment: RiskAssessment
    causal_chain: list[CausalLink]
    metadata: ReportMetadata

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
