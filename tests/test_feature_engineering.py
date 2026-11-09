import numpy as np
from sklearn.model_selection import StratifiedKFold

from feature_engineering import train_only_community_density
from splits import RANDOM_STATE, split_indices


def setup(n=5_000, n_comm=40, seed=1):
    rng = np.random.default_rng(seed)
    community = rng.integers(0, n_comm, n)
    y = (rng.random(n) < 0.05).astype(int)
    train, val, test = split_indices(y)
    return community, y, train, val, test


def test_training_values_come_from_other_folds_only():
    community, y, train, _, _ = setup()
    density = train_only_community_density(community, y, train)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fit_pos, hold_pos in folds.split(train, y[train]):
        source = train[fit_pos]  # excludes the held-out accounts themselves
        for i in train[hold_pos][:10]:
            same = source[community[source] == community[i]]
            assert density[i] == y[same].mean()


def test_val_test_labels_never_used():
    community, y, train, val, test = setup()
    base = train_only_community_density(community, y, train)
    scrambled = y.copy()
    held_out = np.concatenate([val, test])
    scrambled[held_out] = 1 - scrambled[held_out]
    assert np.array_equal(train_only_community_density(community, scrambled, train), base)
