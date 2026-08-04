from pydantic import BaseModel


class FeatureAttribution(BaseModel):
    feature: str
    importance: float


class EdgeAttribution(BaseModel):
    source: str
    target: str
    importance: float


class ExplanationResponse(BaseModel):
    account_id: str
    probability: float
    predicted_label: str
    top_features: list[FeatureAttribution]
    top_edges: list[EdgeAttribution]
    cycle_membership_count: int
    cycle_lengths: list[int]
    explain_hops: int
    explain_subgraph_size: int
