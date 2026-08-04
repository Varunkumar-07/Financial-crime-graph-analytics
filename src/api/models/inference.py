"""
Per-request inference helpers: subgraph extraction and GNNExplainer
attribution, reusing the exact functions from explainability.py so API
responses are consistent with the saved explainability report/PNGs.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from explainability import (  # noqa: E402
    find_account_cycles,
    get_viz_subgraph,
    run_gnn_explainer,
)

__all__ = ["find_account_cycles", "get_viz_subgraph", "run_gnn_explainer"]
