from pydantic import BaseModel
from typing import List, Dict, Optional

class Observation(BaseModel):
    task_id: str
    step_count: int
    visible_emails: Optional[List[int]] = None
    code_snippet: Optional[str] = None
    data_sample: Optional[List[Dict]] = None
    logs: Optional[str] = None
    instruction: str

class Action(BaseModel):
    action_type: str
    payload: Dict

class Reward(BaseModel):
    score: float
    reason: str
