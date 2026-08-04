import math
import sys
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from explainability import EXPLAIN_HOPS  # noqa: E402
from torch_geometric.utils import k_hop_subgraph  # noqa: E402

from api.models.inference import find_account_cycles, get_viz_subgraph, run_gnn_explainer  # noqa: E402
from api.models.state import state  # noqa: E402
from api.schemas.account import (  # noqa: E402
    FlaggedAccount,
    FlaggedAccountsResponse,
    KeyFeatures,
    RiskScoreResponse,
)
from api.schemas.explanation import EdgeAttribution, ExplanationResponse, FeatureAttribution  # noqa: E402
from api.schemas.subgraph import SubgraphEdge, SubgraphNode, SubgraphResponse  # noqa: E402

router = APIRouter(prefix="/accounts", tags=["accounts"])


def _require_account(account_id: str):
    if not state.account_exists(account_id):
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found in graph")


def _key_features(account_id: str) -> KeyFeatures:
    row = state.features_df.loc[account_id]
    return KeyFeatures(
        in_degree=row["in_degree"], out_degree=row["out_degree"], total_degree=row["total_degree"],
        betweenness=row["betweenness"], community_id=int(row["community_id"]),
        community_size=row["community_size"], community_illicit_density=row["community_illicit_density"],
        cycle_count=row["cycle_count"], cycle_flag=row["cycle_flag"], total_sent=row["total_sent"],
        total_received=row["total_received"], transaction_count=row["transaction_count"],
        avg_transaction_amount=row["avg_transaction_amount"], in_out_amount_ratio=row["in_out_amount_ratio"],
    )


# sort_by (API contract) -> flagged_df column (internal name).
_SORT_COLUMNS = {
    "risk_score": "probability",
    "degree": "total_degree",
    "community_illicit_pct": "community_illicit_density",
    "cycle_count": "cycle_count",
}


# NOTE: /accounts/flagged is registered before /accounts/{account_id}/... routes.
# It doesn't actually collide (all account_id routes require a suffix like
# /risk-score), but ordering it first is the safer convention.
@router.get("/flagged", response_model=FlaggedAccountsResponse)
def flagged_accounts(
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(20, ge=1, le=200, description="Results per page (max 200)"),
    sort_by: Literal["risk_score", "degree", "community_illicit_pct", "cycle_count"] = Query(
        "risk_score", description="Sorts the FULL flagged set (not just the current page) before pagination"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc"),
):
    # Sorts all ~12.7K flagged accounts (not the full 515K graph — flagged_df
    # is pre-filtered to proba >= threshold at startup) on every request, then
    # slices the requested page out of the sorted result. kind="mergesort" is
    # stable, so tied values keep a consistent relative order across
    # requests — without that, paginating through a run of ties (e.g. many
    # accounts with cycle_count=0) could show accounts out of order or
    # skip/duplicate them between page loads, since pandas' default
    # quicksort isn't guaranteed stable.
    column = _SORT_COLUMNS[sort_by]
    sorted_df = state.flagged_df.sort_values(column, ascending=(sort_order == "asc"), kind="mergesort")

    total_flagged = len(sorted_df)
    total_pages = max(1, math.ceil(total_flagged / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = sorted_df.iloc[start:end]

    accounts = [
        FlaggedAccount(
            account_id=row.account,
            probability=row.probability,
            predicted_label="illicit",
            total_degree=row.total_degree,
            community_illicit_density=row.community_illicit_density,
            cycle_count=row.cycle_count,
        )
        for row in page_df.itertuples()
    ]
    return FlaggedAccountsResponse(
        page=page, page_size=page_size, total_flagged=total_flagged, total_pages=total_pages,
        threshold=state.threshold, sort_by=sort_by, sort_order=sort_order, accounts=accounts,
    )


@router.get("/{account_id}/risk-score", response_model=RiskScoreResponse)
def risk_score(account_id: str):
    _require_account(account_id)
    proba = state.get_probability(account_id)
    true_label_int = state.get_true_label(account_id)
    return RiskScoreResponse(
        account_id=account_id,
        probability=proba,
        predicted_label="illicit" if proba >= state.threshold else "licit",
        threshold=state.threshold,
        true_label="illicit" if true_label_int == 1 else "licit",
        key_features=_key_features(account_id),
    )


@router.get("/{account_id}/subgraph", response_model=SubgraphResponse)
def subgraph(account_id: str):
    _require_account(account_id)
    G_sub, radius_used = get_viz_subgraph(account_id, state.data, state.G_multi)
    center_community = int(state.features_df.loc[account_id, "community_id"])

    nodes = []
    for n in G_sub.nodes():
        row = state.features_df.loc[n]
        nodes.append(SubgraphNode(
            account_id=n,
            is_center=(n == account_id),
            is_illicit_labeled=bool(row["is_illicit"] == 1),
            community_id=int(row["community_id"]),
            same_community_as_center=(int(row["community_id"]) == center_community),
            total_degree=row["total_degree"],
            cycle_flag=bool(row["cycle_flag"] == 1),
        ))

    # Collapse the MultiDiGraph's parallel edges into one entry per
    # (source, target) pair, counting transactions and flagging illicit.
    collapsed = {}
    for u, v, edata in G_sub.edges(data=True):
        key = (u, v)
        is_ill = edata.get("is_laundering") == 1
        if key in collapsed:
            collapsed[key]["count"] += 1
            collapsed[key]["is_illicit"] = collapsed[key]["is_illicit"] or is_ill
        else:
            collapsed[key] = {"count": 1, "is_illicit": is_ill}

    edges = [
        SubgraphEdge(source=u, target=v, is_illicit=info["is_illicit"], transaction_count=info["count"])
        for (u, v), info in collapsed.items()
    ]

    return SubgraphResponse(
        account_id=account_id, hops=radius_used, node_count=len(nodes), edge_count=len(edges),
        nodes=nodes, edges=edges,
    )


@router.get("/{account_id}/explanation", response_model=ExplanationResponse)
def explanation(account_id: str):
    _require_account(account_id)
    proba = state.get_probability(account_id)

    edge_importance, feat_importance = run_gnn_explainer(account_id, state.model, state.x, state.data)

    idx = state.account_to_idx[account_id]
    subset, _, _, _ = k_hop_subgraph(idx, num_hops=EXPLAIN_HOPS, edge_index=state.data.edge_index, relabel_nodes=True)

    top_feat_idx = feat_importance.argsort()[::-1][:5]
    top_features = [
        FeatureAttribution(feature=state.data.feature_cols[i], importance=float(feat_importance[i]))
        for i in top_feat_idx
    ]

    top_edges_sorted = sorted(edge_importance.items(), key=lambda kv: -kv[1])[:5]
    top_edges = [
        EdgeAttribution(source=u, target=v, importance=float(imp))
        for (u, v), imp in top_edges_sorted
    ]

    account_cycles = find_account_cycles(account_id, state.all_cycles)
    cycle_lengths = sorted(len(c) for c in account_cycles)

    return ExplanationResponse(
        account_id=account_id,
        probability=proba,
        predicted_label="illicit" if proba >= state.threshold else "licit",
        top_features=top_features,
        top_edges=top_edges,
        cycle_membership_count=len(account_cycles),
        cycle_lengths=cycle_lengths,
        explain_hops=EXPLAIN_HOPS,
        explain_subgraph_size=len(subset),
    )
