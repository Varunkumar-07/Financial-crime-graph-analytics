"""
Single source of truth for the account-level train/validation/test split.

Every stage that needs the split (feature engineering, classical
baselines, GraphSAGE training, explainability, the API) imports it from
here, so all of them partition the 515,080 accounts identically:

  - 20% test (stratified on is_illicit) — used only for final metrics
  - 10% validation (12.5% of the remaining 80%, stratified) — used for
    early stopping and decision-threshold selection
  - 70% train — the only labels a model or a label-derived feature
    may see

Indices refer to row positions in data/processed/account_features.csv,
which is also the node order of the PyG Data object.
"""

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE_OF_REMAINDER = 0.125  # 0.125 * 0.8 = 0.1 of the full set


def split_indices(y: np.ndarray):
    """Return (train_idx, val_idx, test_idx) as integer arrays."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_SIZE_OF_REMAINDER, stratify=y[train_idx],
        random_state=RANDOM_STATE,
    )
    return train_idx, val_idx, test_idx
