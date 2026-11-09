"""
Shared train/val/test split used by every stage (70/10/20, stratified).
- test: final metrics only
- val: early stopping and threshold selection
- train: the only labels models/features may see

Indices are row positions in account_features.csv (= PyG node order).
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
