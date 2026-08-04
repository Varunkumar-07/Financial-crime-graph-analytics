from pydantic import BaseModel, Field


class SubgraphNode(BaseModel):
    account_id: str
    is_center: bool
    is_illicit_labeled: bool
    community_id: int
    same_community_as_center: bool
    total_degree: float
    cycle_flag: bool


class SubgraphEdge(BaseModel):
    source: str
    target: str
    is_illicit: bool
    transaction_count: int = 1


class SubgraphResponse(BaseModel):
    account_id: str
    hops: int = Field(..., description="Actual hop radius used (2 if the neighborhood is small enough, else 1)")
    node_count: int
    edge_count: int
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]
