"""
Checks how much the reported scores depend on three choices:
  1. community_illicit_density (uses neighbours' labels)
  2. random split (a time-based split is closer to real use)
  3. one validation set for both early stopping and the threshold

Reuses the exact training code from train_baseline.py / train_gnn.py.
XGBoost and GraphSAGE run in separate processes (xgboost and torch
crash when loaded together on macOS).
Writes notebooks/ablation_summary.md and data/processed/ablation_metrics.json.

Usage: python src/ablation.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import PROJECT_ROOT, TARGET, f2_optimal_threshold, get_feature_cols, load_features, load_transactions
from feature_engineering import train_only_community_density
from splits import RANDOM_STATE, split_indices

REPORT_MD = PROJECT_ROOT / "notebooks" / "ablation_summary.md"
METRICS_JSON = PROJECT_ROOT / "data" / "processed" / "ablation_metrics.json"
DENSITY = "community_illicit_density"

LABELS = {
    "reported": "As reported",
    "no_community_density": "No community_illicit_density",
    "temporal_split": "Time-based split",
    "separate_val": "Separate early-stop / threshold sets",
}


def score(y_test, test_proba, y_thresh, thresh_proba):
    thresh = f2_optimal_threshold(y_thresh, thresh_proba)[0]
    pred = (test_proba >= thresh).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_test, test_proba)),
        "auc_pr": float(average_precision_score(y_test, test_proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def temporal_split(accounts):
    """70/10/20 split by each account's first transaction time."""
    df = load_transactions(parse_timestamps=True)
    first_sent = df.groupby("From Account")["Timestamp"].min()
    first_recv = df.groupby("To Account")["Timestamp"].min()
    first_seen = pd.concat([first_sent, first_recv]).groupby(level=0).min().reindex(accounts)
    assert first_seen.notna().all()
    order = np.argsort(first_seen.values, kind="stable")
    n = len(order)
    return order[: int(0.7 * n)], order[int(0.7 * n): int(0.8 * n)], order[int(0.8 * n):]


def build_setups(df):
    """Every setup: (train, stop, thresh, test) indices + optional density override."""
    y = df[TARGET].values
    train, val, test = split_indices(y)
    t_train, t_val, t_test = temporal_split(df["account"].tolist())
    stop, thresh = train_test_split(val, test_size=0.5, stratify=y[val], random_state=RANDOM_STATE)
    return {
        "reported": dict(idx=(train, val, val, test)),
        "no_community_density": dict(idx=(train, val, val, test), drop_density=True),
        "temporal_split": dict(
            idx=(t_train, t_val, t_val, t_test),
            density=train_only_community_density(df["community_id"].values, y, t_train),
        ),
        "separate_val": dict(idx=(train, stop, thresh, test), gnn_only=True),
    }


def apply_setup(X, feature_cols, setup):
    X = X.copy()
    d = feature_cols.index(DENSITY)
    if "density" in setup:
        X[:, d] = setup["density"]
    if setup.get("drop_density"):
        keep = [i for i in range(X.shape[1]) if i != d]
        return X[:, keep], [feature_cols[i] for i in keep]
    return X, feature_cols


def run_xgboost(setups, out_path):
    from train_baseline import make_xgboost

    df = load_features()
    feature_cols = get_feature_cols(df)
    X_all, y = df[feature_cols].values, df[TARGET].values
    results = {}
    for name, setup in setups.items():
        if setup.get("gnn_only"):
            continue
        print(f"  XGBoost: {LABELS[name]}", flush=True)
        train, _, thresh, test = setup["idx"]
        X, _ = apply_setup(X_all, feature_cols, setup)
        n_pos = y[train].sum()
        model = make_xgboost((len(train) - n_pos) / n_pos)
        model.fit(X[train], y[train])
        proba = model.predict_proba(X)[:, 1]
        results[name] = score(y[test], proba[test], y[thresh], proba[thresh])
    Path(out_path).write_text(json.dumps(results))


def run_graphsage(setups, out_path):
    import torch
    import torch.nn.functional as F

    from train_gnn import PYG_DATA_PATH, fit_graphsage, preprocess_features

    data = torch.load(PYG_DATA_PATH, weights_only=False)
    y = data.y.numpy()
    x_all = data.x.numpy()
    results = {}
    for name, setup in setups.items():
        train, stop, thresh, test = setup["idx"]
        x_np, cols = apply_setup(x_all, data.feature_cols, setup)
        x = torch.tensor(preprocess_features(x_np, cols, train), dtype=torch.float)
        model, _, stats = fit_graphsage(x, data.edge_index, data.y, train, stop, verbose=False)
        with torch.no_grad():
            proba = F.softmax(model(x, data.edge_index), dim=1)[:, 1].numpy()
        print(f"  GraphSAGE: {LABELS[name]} ({stats['epochs']} epochs, {stats['train_time']:.0f}s)", flush=True)
        results[name] = score(y[test], proba[test], y[thresh], proba[thresh])
    Path(out_path).write_text(json.dumps(results))


def write_report(results, test_rate_temporal, runtime):
    lines = [
        "# Ablation Summary",
        "",
        "How much the reported test scores depend on three methodology choices.",
        "Thresholds are always picked on validation data (max F2).",
        "",
        "| Setup | Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in LABELS.items():
        for model, name in (("xgboost", "XGBoost"), ("graphsage", "GraphSAGE")):
            m = results[model].get(key)
            if m:
                lines.append(f"| {label} | {name} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} | "
                             f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    lines += [
        "",
        "Notes:",
        "- Time-based split: accounts ordered by first transaction; oldest 70% train, next 10% val, "
        f"newest 20% test (test positive rate {100 * test_rate_temporal:.2f}%). "
        "community_illicit_density is recomputed from the new training labels; other features "
        "still use the full period, so this is a partial check.",
        "- Separate sets: validation split in half, one half for early stopping, the other for "
        "the threshold (GraphSAGE only; XGBoost has no early stopping).",
        "",
        f"Total runtime: {runtime:.0f}s",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    t_start = time.time()
    df = load_features()
    print("Building splits...", flush=True)
    setups = build_setups(df)

    with tempfile.TemporaryDirectory() as tmp:
        setups_path = Path(tmp) / "setups.npy"
        np.save(setups_path, setups, allow_pickle=True)
        results = {}
        for part in ("xgboost", "graphsage"):
            out = Path(tmp) / f"{part}.json"
            subprocess.run([sys.executable, __file__, part, str(setups_path), str(out)], check=True)
            results[part] = json.loads(out.read_text())

    METRICS_JSON.write_text(json.dumps(results, indent=2))
    test_rate = float(df[TARGET].values[setups["temporal_split"]["idx"][3]].mean())
    write_report(results, test_rate, time.time() - t_start)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        part, setups_file, out_file = sys.argv[1:]
        loaded = np.load(setups_file, allow_pickle=True).item()
        {"xgboost": run_xgboost, "graphsage": run_graphsage}[part](loaded, out_file)
    else:
        main()
