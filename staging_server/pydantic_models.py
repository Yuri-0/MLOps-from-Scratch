from pydantic import BaseModel
from typing import Any, Literal


class setup_request(BaseModel):
    experiment_id: str
    name: str
    user_id: str
    status: Literal['RUNNING', 'DONE']
    start_time: float


class params_request(BaseModel):
    param_name: str
    value: Any
    experiment_id: str


class metrics_request(BaseModel):
    metric_name: str
    value: int | float
    timestamp: float
    experiment_id: str
    step: int | None


class tags_request(BaseModel):
    tags: str
    value: Any
    experiment_id: str


class load_request(BaseModel):
    name: str
    user_id: str
    version: int | None


class delete_request(BaseModel):
    name: str
    user_id: str
    version: int
    deleted_time: float
