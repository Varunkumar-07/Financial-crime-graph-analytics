from typing import Optional

from pydantic import BaseModel, Field


class KeyFeatures(BaseModel):
    in_degree: float
    out_degree: float
    total_degree: float
    betweenness: float
    community_id: int
    community_size: float
    community_illicit_density: float
    cycle_count: float
    cycle_flag: float
    total_sent: float
    total_received: float
    transaction_count: float
    avg_transaction_amount: float
    in_out_amount_ratio: float


class RiskScoreResponse(BaseModel):
    account_id: str
    probability: float = Field(..., description="GraphSAGE predicted probability of being illicit")
    predicted_label: str = Field(..., description="'illicit' or 'licit' at the decision threshold")
    threshold: float = Field(..., description="F2-optimal decision threshold used for predicted_label")
    true_label: Optional[str] = Field(
        None, description="Ground-truth label if known ('illicit'/'licit'); null if unlabeled in this dataset"
    )
    key_features: KeyFeatures


class FlaggedAccount(BaseModel):
    account_id: str
    probability: float
    predicted_label: str


class FlaggedAccountsResponse(BaseModel):
    page: int
    page_size: int
    total_flagged: int
    total_pages: int
    threshold: float
    accounts: list[FlaggedAccount]
