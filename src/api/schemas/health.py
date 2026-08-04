from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_accounts: int
    n_cycles_cached: int
    threshold: float
