from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Action(BaseModel):
    action_type: str = Field(..., description="E.g., 'search_kb', 'check_order', 'issue_refund', 'draft_reply', 'route_ticket'")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action arguments, e.g., {'query': 'refund policy'}")

class Observation(BaseModel):
    current_ticket: Optional[Dict[str, Any]]
    action_result: str = Field(..., description="Result of the last action taken")
    system_messages: str = Field(..., description="Warnings or errors")

class Reward(BaseModel):
    step_reward: float
    total_score: float
    is_done: bool
    info: Dict[str, Any]