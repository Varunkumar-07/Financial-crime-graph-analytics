"""
Loads the trained GraphSAGE model, the PyG graph, the original
transaction MultiDiGraph, the feature table, and the cached cycle set
ONCE into a module-level AppState singleton, reused across all
requests. Nothing here re-loads or re-computes per-request except
GNNExplainer (which is inherently query-specific — see explanation.py).

Reuses the exact preprocessing, split, and threshold logic from
train_gnn.py / explainability.py so API predictions are identical to
the ones reported in earlier phases (same random_state=42 split).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

import graph_analysis as ga  # noqa: E402
from explainability import get_cycles_cached  # noqa: E402
from train_gnn import GraphSAGE, RANDOM_STATE, f2_optimal_threshold, preprocess_features  # noqa: E402

PROJECT_ROOT = SRC_DIR.parent


class AppState:
    def __init__(self):
        self.loaded = False

    def load(self):
        t0 = time.time()
        print("[startup] Loading PyG data...")
        self.data = torch.load(PROJECT_ROOT / "data" / "processed" / "pyg_data.pt", weights_only=False)
        ckpt = torch.load(PROJECT_ROOT / "data" / "processed" / "graphsage_baseline.pt", weights_only=False)

        idx = np.arange(self.data.num_nodes)
        y_np = self.data.y.numpy()
        train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y_np, random_state=RANDOM_STATE)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.125, stratify=y_np[train_idx], random_state=RANDOM_STATE
        )

        print("[startup] Preprocessing features and loading model...")
        x_np = preprocess_features(self.data.x.numpy().copy(), self.data.feature_cols, train_idx)
        self.x = torch.tensor(x_np, dtype=torch.float)

        self.model = GraphSAGE(ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print("[startup] Running full-graph inference once...")
        with torch.no_grad():
            out = self.model(self.x, self.data.edge_index)
            self.proba_all = F.softmax(out, dim=1)[:, 1].numpy()

        self.threshold, _ = f2_optimal_threshold(y_np[val_idx], self.proba_all[val_idx])
        self.y_np = y_np

        # Live eval metrics for the model actually loaded, computed from data
        # already in memory (no extra inference) — this is what /model/info
        # serves, so the frontend never shows numbers that can drift from the
        # model actually on disk (see notebooks/integration_test_summary.md
        # for the incident that prompted this).
        self.auc_roc = float(roc_auc_score(y_np[test_idx], self.proba_all[test_idx]))
        self.auc_pr = float(average_precision_score(y_np[test_idx], self.proba_all[test_idx]))
        self.n_test_accounts = int(len(test_idx))

        print("[startup] Loading feature table and original graph...")
        self.features_df = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "account_features.csv"
        ).set_index("account")

        self.account_to_idx = {a: i for i, a in enumerate(self.data.accounts)}

        print("[startup] Building flagged accounts table (proba >= threshold)...")
        flagged_mask = self.proba_all >= self.threshold
        flagged_idx = np.where(flagged_mask)[0]
        flagged_account_ids = [self.data.accounts[i] for i in flagged_idx]
        # One row per flagged account, columns for every sortable field the
        # /accounts/flagged endpoint supports — sorted server-side, on the
        # full flagged set, on every request (see accounts.py). Kept as a
        # DataFrame (not pre-sorted at startup) since 12,748 rows is cheap
        # to re-sort per request and this avoids maintaining N precomputed
        # orderings that could drift out of sync.
        flagged_df = self.features_df.loc[
            flagged_account_ids, ["total_degree", "community_illicit_density", "cycle_count"]
        ].copy()
        flagged_df["account"] = flagged_account_ids
        flagged_df["probability"] = self.proba_all[flagged_idx]
        self.flagged_df = flagged_df.reset_index(drop=True)

        print("[startup] Loading original MultiDiGraph...")
        self.G_multi = ga.load_graph()

        print("[startup] Loading cached cycle detection results...")
        self.all_cycles, _ = get_cycles_cached()

        self.loaded = True
        print(f"[startup] Ready in {time.time()-t0:.1f}s. "
              f"{self.data.num_nodes:,} accounts, {len(self.all_cycles):,} cycles cached, "
              f"threshold={self.threshold:.4f}")

    def account_exists(self, account_id: str) -> bool:
        return account_id in self.account_to_idx

    def get_probability(self, account_id: str) -> float:
        return float(self.proba_all[self.account_to_idx[account_id]])

    def get_true_label(self, account_id: str):
        return int(self.y_np[self.account_to_idx[account_id]])


state = AppState()
