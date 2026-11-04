"""
Train and evaluate a GraphSAGE node-classification baseline on the
transaction graph, directly comparable to the XGBoost/RF baselines in
train_baseline.py (same metrics, same 80/20 stratified split logic —
here expressed as train/test masks over one graph rather than separate
row splits, the standard PyG node-classification setup).

Full-batch training (not NeighborLoader): timed at ~0.85s/epoch on
CPU for this graph size (515K nodes, 1.66M edges), comfortably under
the 10-15 min/epoch threshold that would have required sampling.
MPS was tested and found slightly SLOWER than CPU (kernel-dispatch
overhead dominates at this graph size), so CPU is used.

Class imbalance handled via a weighted cross-entropy loss (weight
inversely proportional to class frequency in the training set) —
the GNN analogue of scale_pos_weight/class_weight='balanced' used in
the classical baselines.

Run from the project root with the venv activated:

    python src/train_gnn.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import RANDOM_STATE, split_indices  # noqa: E402,F401 (RANDOM_STATE re-exported)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    """log1p the heavily right-skewed count/dollar columns, then z-score
    standardize every column using TRAIN-set statistics only (applied to
    all rows). Unlike the tree-based baselines, GraphSAGE is scale-sensitive:
    the first training attempt without this step produced loss values in
    the tens of millions and AUC-ROC ~0.50 (no better than random) because
    raw features like total_sent span into the hundreds of millions.
    """
    x_np = x_np.copy()
    for col in LOG1P_COLS:
        j = feature_cols.index(col)
        x_np[:, j] = np.log1p(x_np[:, j])

    # in_out_amount_ratio carries a -1 sentinel (accounts that never sent,
    # see feature_engineering.py) alongside otherwise non-negative, skewed
    # ratios. log1p only the valid (non-sentinel) values; leave -1 as a
    # distinct value so standardization keeps it separable from real ratios.
    ratio_j = feature_cols.index("in_out_amount_ratio")
    ratio_vals = x_np[:, ratio_j]
    valid = ratio_vals != -1
    x_np[valid, ratio_j] = np.log1p(ratio_vals[valid])

    train_mean = x_np[train_idx].mean(axis=0)
    train_std = x_np[train_idx].std(axis=0)
    train_std[train_std == 0] = 1.0
    return (x_np - train_mean) / train_std


def f2_optimal_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = 5 * precision * recall / (4 * precision + recall)
    f2 = np.nan_to_num(f2)
    best_idx = np.argmax(f2)
    return thresholds[best_idx], f2[best_idx]


def main() -> None:
    t_start = time.time()
    torch.manual_seed(RANDOM_STATE)

    print("Loading PyG data...")
    data = torch.load(PYG_DATA_PATH, weights_only=False)
    print(f"  {data}")

    # Shared 70/10/20 split (splits.py), identical to train_baseline.py,
    # expressed as masks over the single graph (standard PyG setup). The
    # validation set drives early stopping and threshold selection, so
    # test-set metrics aren't used to tune anything.
    y_np = data.y.numpy()
    train_idx, val_idx, test_idx = split_indices(y_np)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  Train: {train_mask.sum().item():,} ({y_np[train_idx].mean()*100:.4f}% pos)")
    print(f"  Val:   {val_mask.sum().item():,} ({y_np[val_idx].mean()*100:.4f}% pos)")
    print(f"  Test:  {test_mask.sum().item():,} ({y_np[test_idx].mean()*100:.4f}% pos)")

    x_np = preprocess_features(data.x.numpy().copy(), data.feature_cols, train_idx)

    device = torch.device("cpu")  # see module docstring: MPS timed slower at this scale
    x = torch.tensor(x_np, dtype=torch.float, device=device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    n_pos = y[train_mask].sum().item()
    n_neg = train_mask.sum().item() - n_pos
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
    print(f"  Class weights (loss): licit=1.0, illicit={class_weights[1].item():.2f}")

    model = GraphSAGE(data.num_node_features, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_aucpr = -1.0
    best_state = None
    epochs_without_improvement = 0

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
            # Use val_idx (not val_mask) to index `out`, matching the exact row
            # order of y_np[val_idx] — boolean-mask indexing returns rows in
            # ascending node-index order, which does NOT match val_idx's order
            # (train_test_split shuffles), so mixing the two silently
            # mispairs predictions with labels. See module notes.
            val_proba = F.softmax(out[val_idx], dim=1)[:, 1].cpu().numpy()
            val_aucpr = average_precision_score(y_np[val_idx], val_proba)

        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: loss={loss.item():.4f}  val_AUC-PR={val_aucpr:.4f}  "
                  f"best={best_val_aucpr:.4f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no val AUC-PR improvement for {PATIENCE} epochs)")
            break

    train_time = time.time() - t0
    n_epochs_run = epoch
    print(f"Training done in {train_time:.1f}s ({train_time/n_epochs_run:.3f}s/epoch, {n_epochs_run} epochs)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        # See note above: index with test_idx/val_idx (not test_mask/val_mask)
        # so prediction rows and label rows stay in the same order.
        test_proba = F.softmax(out[test_idx], dim=1)[:, 1].cpu().numpy()

    y_test = y_np[test_idx]
    auc_roc = roc_auc_score(y_test, test_proba)
    auc_pr = average_precision_score(y_test, test_proba)

    # Threshold chosen on the VALIDATION set (not test), then applied to test —
    # avoids the test-set threshold leakage flagged as a caveat in train_baseline.py.
    with torch.no_grad():
        val_proba_final = F.softmax(out[val_idx], dim=1)[:, 1].cpu().numpy()
    thresh, val_f2 = f2_optimal_threshold(y_np[val_idx], val_proba_final)

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
        f"{class_weights[1].item():.2f} (train neg/pos ratio) — the GNN analogue of "
        "scale_pos_weight/class_weight='balanced' in the classical baselines",
        f"- Optimizer: Adam, lr={LR}, weight_decay={WEIGHT_DECAY}",
        f"- Trained {n_epochs_run} epochs (early stopping, patience={PATIENCE}) in "
        f"{train_time:.1f}s ({train_time/n_epochs_run:.3f}s/epoch) on CPU",
        f"- Best validation AUC-PR: {best_val_aucpr:.4f}",
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
