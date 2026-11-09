import numpy as np

from splits import split_indices


def make_labels(n=10_000, pos_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < pos_rate).astype(int)


def test_sizes_and_disjoint():
    y = make_labels()
    tr, va, te = split_indices(y)
    assert len(tr) + len(va) + len(te) == len(y)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert abs(len(te) / len(y) - 0.2) < 0.01
    assert abs(len(va) / len(y) - 0.1) < 0.01


def test_stratified():
    y = make_labels()
    for idx in split_indices(y):
        assert abs(y[idx].mean() - y.mean()) < 0.005


def test_deterministic():
    y = make_labels()
    a, b = split_indices(y), split_indices(y)
    for x, z in zip(a, b):
        assert np.array_equal(x, z)
