from pydantic import BaseModel
from typing import List, Optional, Dict

class Service(BaseModel):
    name: str
    status: str        # "healthy" | "degraded" | "down"
    cpu: float
    memory: float
    error_rate: float

class Alert(BaseModel):
    id: str
    service: str
    severity: str      # "low" | "medium" | "high" | "critical"
    message: str
    timestamp: str

class Observation(BaseModel):
    step: int
    services: List[Service]
    alerts: List[Alert]
    recent_logs: Dict[str, List[str]]
    last_action_result: Optional[str]
    actions_remaining: int
    done: bool

class Action(BaseModel):
    name: str
    target: Optional[str] = None
    message: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str
    cumulative: float