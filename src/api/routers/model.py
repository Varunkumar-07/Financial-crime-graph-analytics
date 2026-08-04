from fastapi import APIRouter

from api.models.state import state
from api.schemas.model_info import ModelInfoResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name="GraphSAGE",
        auc_roc=state.auc_roc,
        auc_pr=state.auc_pr,
        threshold=state.threshold,
        n_test_accounts=state.n_test_accounts,
    )
