import numpy as np
import pandas as pd

from common import EXCLUDED_COLS, f2_optimal_threshold, get_feature_cols


def test_f2_threshold_separable():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thresh, f2, precision, recall = f2_optimal_threshold(y, p)
    assert thresh == 0.8
    assert f2 == 1.0 and precision == 1.0 and recall == 1.0


def test_feature_cols_drop_id_and_label():
    df = pd.DataFrame(columns=EXCLUDED_COLS + ["in_degree", "total_sent"])
    assert get_feature_cols(df) == ["in_degree", "total_sent"]
