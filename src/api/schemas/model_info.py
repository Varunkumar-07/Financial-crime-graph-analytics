from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Name of the model currently serving predictions")
    auc_roc: float = Field(..., description="Test-set AUC-ROC, computed live from the loaded model — never hardcoded")
    auc_pr: float = Field(..., description="Test-set AUC-PR (average precision), computed live from the loaded model")
    threshold: float
    n_test_accounts: int
