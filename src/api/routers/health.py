from fastapi import APIRouter

from api.models.state import state
from api.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state.loaded else "loading",
        model_loaded=state.loaded,
        n_accounts=state.data.num_nodes if state.loaded else 0,
        n_cycles_cached=len(state.all_cycles) if state.loaded else 0,
        threshold=state.threshold if state.loaded else -1.0,
    )
