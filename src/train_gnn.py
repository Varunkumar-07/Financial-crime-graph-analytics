"""
Trains and evaluates GraphSAGE, comparable to train_baseline.py
(same split and metrics).

- Full-batch on CPU (~0.85s/epoch; MPS was slower).
- Class imbalance handled with weighted cross-entropy.

Usage: python src/train_gnn.py
"""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import SAGEConv

from common import PROJECT_ROOT, f2_optimal_threshold
from splits import RANDOM_STATE, split_indices

PYG_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pyg_data.pt"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "graphsage_baseline.pt"
REPORT_MD = PROJECT_ROOT / "notebooks" / "gnn_baseline_summary.md"
BASELINE_METRICS_JSON = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
GNN_METRICS_JSON = PROJECT_ROOT / "data" / "processed" / "gnn_metrics.json"

HIDDEN_DIM = 64
N_EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
PATIENCE = 20  # early stopping on val AUC-PR


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)


LOG1P_COLS = [
    "in_degree", "out_degree", "total_degree", "betweenness", "community_size",
    "cycle_count", "total_sent", "total_received", "transaction_count",
    "avg_transaction_amount",
]


def preprocess_features(x_np: np.ndarray, feature_cols: list, train_idx: np.ndarray) -> np.ndarray:
    """log1p skewed columns, then standardize using train-set stats.
    GraphSAGE needs scaled inputs (without this, AUC-ROC was ~0.50).
    """
    x_np = x_np.copy()
    for col in LOG1P_COLS:
        j = feature_cols.index(col)
        x_np[:, j] = np.log1p(x_np[:, j])

    # Keep the -1 "never sent" value; log1p only real ratios.
    ratio_j = feature_cols.index("in_out_amount_ratio")
    ratio_vals = x_np[:, ratio_j]
    valid = ratio_vals != -1
    x_np[valid, ratio_j] = np.log1p(ratio_vals[valid])

    train_mean = x_np[train_idx].mean(axis=0)
    train_std = x_np[train_idx].std(axis=0)
    train_std[train_std == 0] = 1.0
    return (x_np - train_mean) / train_std


def fit_graphsage(x, edge_index, y, train_idx, val_idx, verbose=True):
    """Full-batch training with early stopping on val AUC-PR.
    Returns (model with best weights, best_state, stats dict).
    """
    torch.manual_seed(RANDOM_STATE)
    y_np = y.numpy()
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True

    n_pos = y[train_mask].sum().item()
    n_neg = train_mask.sum().item() - n_pos
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float)
    if verbose:
        print(f"  Class weights (loss): licit=1.0, illicit={class_weights[1].item():.2f}")

    # CPU only: MPS was slower here
    model = GraphSAGE(x.shape[1], hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_aucpr = -1.0
    best_state = None
    epochs_without_improvement = 0

    if verbose:
        print(f"Training GraphSAGE for up to {N_EPOCHS} epochs (patience={PATIENCE})...")
    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            # Index with val_idx, not a mask, so rows line up with y_np[val_idx].
            val_proba = F.softmax(out[val_idx], dim=1)[:, 1].numpy()
            val_aucpr = average_precision_score(y_np[val_idx], val_proba)

        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}: loss={loss.item():.4f}  val_AUC-PR={val_aucpr:.4f}  "
                  f"best={best_val_aucpr:.4f}")

        if epochs_without_improvement >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no val AUC-PR improvement for {PATIENCE} epochs)")
            break

    train_time = time.time() - t0
    if verbose:
        print(f"Training done in {train_time:.1f}s ({train_time/epoch:.3f}s/epoch, {epoch} epochs)")

    model.load_state_dict(best_state)
    model.eval()
    stats = {"epochs": epoch, "train_time": train_time, "best_val_aucpr": best_val_aucpr,
             "illicit_weight": class_weights[1].item()}
    return model, best_state, stats


def main() -> None:
    t_start = time.time()

    print("Loading PyG data...")
    data = torch.load(PYG_DATA_PATH, weights_only=False)
    print(f"  {data}")

    # Same split as train_baseline.py; val is used for early stopping and threshold.
    y_np = data.y.numpy()
    train_idx, val_idx, test_idx = split_indices(y_np)

    print(f"  Train: {len(train_idx):,} ({y_np[train_idx].mean()*100:.4f}% pos)")
    print(f"  Val:   {len(val_idx):,} ({y_np[val_idx].mean()*100:.4f}% pos)")
    print(f"  Test:  {len(test_idx):,} ({y_np[test_idx].mean()*100:.4f}% pos)")

    x_np = preprocess_features(data.x.numpy().copy(), data.feature_cols, train_idx)
    x = torch.tensor(x_np, dtype=torch.float)

    model, best_state, stats = fit_graphsage(x, data.edge_index, data.y, train_idx, val_idx)
    n_epochs_run, train_time = stats["epochs"], stats["train_time"]
    with torch.no_grad():
        out = model(x, data.edge_index)
        # Index with idx arrays (not masks) to keep rows aligned with labels.
        test_proba = F.softmax(out[test_idx], dim=1)[:, 1].cpu().numpy()

    y_test = y_np[test_idx]
    auc_roc = roc_auc_score(y_test, test_proba)
    auc_pr = average_precision_score(y_test, test_proba)

    # Pick threshold on validation, apply to test.
    with torch.no_grad():
        val_proba_final = F.softmax(out[val_idx], dim=1)[:, 1].cpu().numpy()
    thresh, val_f2, _, _ = f2_optimal_threshold(y_np[val_idx], val_proba_final)

    y_pred_default = (test_proba >= 0.5).astype(int)
    y_pred_chosen = (test_proba >= thresh).astype(int)

    def metrics_at(y_pred):
        return {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

    default_metrics = metrics_at(y_pred_default)
    chosen_metrics = metrics_at(y_pred_chosen)
    cm = confusion_matrix(y_test, y_pred_chosen)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTest AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")
    print(f"Chosen threshold (from val F2): {thresh:.4f}")
    print(f"Test @ chosen threshold: precision={chosen_metrics['precision']:.4f}, "
          f"recall={chosen_metrics['recall']:.4f}, f1={chosen_metrics['f1']:.4f}")

    torch.save({"model_state": best_state, "hidden_dim": HIDDEN_DIM,
                "in_dim": data.num_node_features}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    GNN_METRICS_JSON.write_text(json.dumps({
        "auc_roc": auc_roc, "auc_pr": auc_pr, "threshold": float(thresh),
        **chosen_metrics, "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "n_test": int(len(test_idx)), "epochs": n_epochs_run,
        "sec_per_epoch": train_time / n_epochs_run,
    }, indent=2))
    baselines = json.loads(BASELINE_METRICS_JSON.read_text())

    lines = [
        "# GNN Baseline Summary (GraphSAGE)",
        "",
        "## Setup",
        f"- Graph: {data.num_nodes:,} nodes, {data.num_edges:,} directed edges "
        "(collapsed multi-edges, made undirected — see build_pyg_data.py)",
        f"- Node features: {data.num_node_features} (same as classical baselines, "
        "excluding account/community_id/is_illicit)",
        "- Feature preprocessing: log1p on heavily right-skewed count/dollar columns, then "
        "z-score standardization using TRAIN-set statistics only. Required: an initial run on "
        "raw (unscaled) features — fine for the tree-based baselines, which are scale-invariant "
        "— produced loss values in the tens of millions and test AUC-ROC 0.5034 (no better than "
        "random) for GraphSAGE, since features like total_sent range into the hundreds of "
        "millions and destabilized training.",
        "- Architecture: 3-layer GraphSAGE, hidden_dim=64, dropout=0.3, ReLU",
        "- Split: shared 70/10/20 train/validation/test split (splits.py, stratified), "
        "identical to the classical baselines; validation drives early stopping and threshold selection — "
        "test set is never used for anything but final reported metrics",
        f"- Class imbalance handling: weighted cross-entropy loss, illicit class weight="
        f"{stats['illicit_weight']:.2f} (train neg/pos ratio) — the GNN analogue of "
        "scale_pos_weight/class_weight='balanced' in the classical baselines",
        f"- Optimizer: Adam, lr={LR}, weight_decay={WEIGHT_DECAY}",
        f"- Trained {n_epochs_run} epochs (early stopping, patience={PATIENCE}) in "
        f"{train_time:.1f}s ({train_time/n_epochs_run:.3f}s/epoch) on CPU",
        f"- Best validation AUC-PR: {stats['best_val_aucpr']:.4f}",
        "",
        "## Test set evaluation",
        f"- AUC-ROC: {auc_roc:.4f}",
        f"- AUC-PR (average precision): {auc_pr:.4f}",
        "",
        f"**At default threshold 0.5:** precision={default_metrics['precision']:.4f}, "
        f"recall={default_metrics['recall']:.4f}, f1={default_metrics['f1']:.4f}",
        "",
        f"**At chosen threshold {thresh:.4f} (F2-optimal on validation set, not test — "
        "avoids the threshold-leakage caveat noted in the classical baseline report):** "
        f"precision={chosen_metrics['precision']:.4f}, recall={chosen_metrics['recall']:.4f}, "
        f"f1={chosen_metrics['f1']:.4f}",
        "",
        "Confusion matrix at chosen threshold:",
        "",
        "| | Predicted licit | Predicted illicit |",
        "|---|---|---|",
        f"| Actual licit | {tn:,} (TN) | {fp:,} (FP) |",
        f"| Actual illicit | {fn:,} (FN) | {tp:,} (TP) |",
        "",
        "## Comparison to classical baselines (same test set)",
        "",
        "| Model | AUC-ROC | AUC-PR |",
        "|---|---|---|",
        f"| XGBoost | {baselines['XGBoost']['auc_roc']:.4f} | {baselines['XGBoost']['auc_pr']:.4f} |",
        f"| Random Forest | {baselines['Random Forest']['auc_roc']:.4f} | "
        f"{baselines['Random Forest']['auc_pr']:.4f} |",
        f"| GraphSAGE | {auc_roc:.4f} | {auc_pr:.4f} |",
        "",
        f"Total runtime: {time.time() - t_start:.1f}s",
        "",
    ]
    summary = "\n".join(lines)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
