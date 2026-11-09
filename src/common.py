"""Shared paths and helpers used across the pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "account_features.csv"

TARGET = "is_illicit"
EXCLUDED_COLS = ["account", "community_id", TARGET]


def load_transactions(parse_timestamps: bool = False) -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # CSV has two "Account" columns; pandas names the second "Account.1".
    df = df.rename(columns={"Account": "From Account", "Account.1": "To Account"})
    if parse_timestamps:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    return df


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    # Accounts that never sent have no ratio; -1 marks them.
    df["in_out_amount_ratio"] = df["in_out_amount_ratio"].fillna(-1)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDED_COLS]


def f2_optimal_threshold(y_true, y_proba):
    """Returns (threshold, f2, precision, recall) at the F2-best threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]  # drop last point (no threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = 5 * precision * recall / (4 * precision + recall)
    f2 = np.nan_to_num(f2)
    best_idx = np.argmax(f2)
    return thresholds[best_idx], f2[best_idx], precision[best_idx], recall[best_idx]
