"""API route tests against a small fake state (no model or data files)."""

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.models.state import state
from api.routers import accounts

FEATURES = ["in_degree", "out_degree", "total_degree", "betweenness", "community_id",
            "community_size", "community_illicit_density", "cycle_count", "cycle_flag",
            "total_sent", "total_received", "transaction_count", "avg_transaction_amount",
            "in_out_amount_ratio", "is_illicit"]


@pytest.fixture
def client(monkeypatch):
    ids = ["A100", "B200", "C300", "D400"]
    features = pd.DataFrame(1.0, index=ids, columns=FEATURES)
    features.loc["A100", "in_out_amount_ratio"] = np.nan  # never sent
    proba = np.array([0.95, 0.90, 0.10, 0.85])
    flagged = pd.DataFrame({
        "account": ["A100", "B200", "D400"],
        "probability": [0.95, 0.90, 0.85],
        "total_degree": [3.0, 1.0, 2.0],
        "community_illicit_density": [0.1, 0.2, 0.3],
        "cycle_count": [0, 0, 1],
    })
    monkeypatch.setattr(state, "features_df", features, raising=False)
    monkeypatch.setattr(state, "flagged_df", flagged, raising=False)
    monkeypatch.setattr(state, "account_to_idx", {a: i for i, a in enumerate(ids)}, raising=False)
    monkeypatch.setattr(state, "proba_all", proba, raising=False)
    monkeypatch.setattr(state, "y_np", np.array([1, 0, 0, 1]), raising=False)
    monkeypatch.setattr(state, "threshold", 0.8, raising=False)
    app = FastAPI()
    app.include_router(accounts.router)
    return TestClient(app)


def test_flagged_sorted_and_paged(client):
    r = client.get("/accounts/flagged", params={"page": 1, "page_size": 2, "sort_by": "degree"}).json()
    assert r["total_flagged"] == 3 and r["total_pages"] == 2
    assert [a["account_id"] for a in r["accounts"]] == ["A100", "D400"]


def test_flagged_search_covers_all_pages(client):
    r = client.get("/accounts/flagged", params={"page_size": 1, "search": "d4"}).json()
    assert r["total_flagged"] == 1
    assert r["accounts"][0]["account_id"] == "D400"


def test_risk_score_null_ratio(client):
    r = client.get("/accounts/A100/risk-score").json()
    assert r["key_features"]["in_out_amount_ratio"] is None
    assert r["predicted_label"] == "illicit" and r["true_label"] == "illicit"


def test_unknown_account_404(client):
    assert client.get("/accounts/NOPE/risk-score").status_code == 404
