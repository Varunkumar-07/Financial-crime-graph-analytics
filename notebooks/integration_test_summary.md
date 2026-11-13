# Integration Test Summary

End-to-end verification that the pipeline (raw data → graph → analysis → features →
models → running API) rebuilds correctly and unattended from a clean state, and that
the full stack (backend + frontend) actually works against a freshly built system —
not just against artifacts accumulated interactively across earlier phases.

> **Historical record.** This describes the pipeline at the time of this test. Since then:
> - The pipeline has **7 steps**: `explainability.py` now runs as Step 6 (rerun whenever
>   GraphSAGE is retrained), and the backend is Step 7.
> - Giant-SCC cycle detection is **exact and deterministic** (rank-ordered depth-limited
>   search, ~0.1 s), not a 120 s wall-clock budget, so cycle counts no longer vary between
>   runs. Total: 5,155 cycles of length 2-6.
> - The frontend header fetches AUC-ROC / AUC-PR live from `GET /model/info`; nothing is
>   hardcoded.
> - CORS origins can be overridden with the `CORS_ALLOWED_ORIGINS` env var.
> - All models now share one stratified 70/10/20 split with validation-chosen thresholds;
>   the metric values quoted below (e.g. AUC-ROC 0.9218) are from older models. Current
>   numbers are in `README.md` and `gnn_baseline_summary.md`.

## What was built

**[scripts/run_pipeline.sh](../scripts/run_pipeline.sh)** — a single idempotent script that runs, in order:
EDA → graph construction → graph analysis → feature engineering → model training →
start backend API. Each stage is skipped if its expected output already exists;
`--retrain` forces the model-training stages to rerun regardless. Full flag/skip-check
details are in the script's own header comment.

**A real bug fixed along the way, before any testing began:** `graph_analysis.py`
(Phase 5) never actually used the cycle-detection disk cache added later in Phase 11
(`explainability.py`) — it always recomputed cycles from scratch (~150s of its ~9min
runtime), and `explainability.py` duplicated the cycle-finding logic independently.
Since this phase's instructions specifically asked for the Graph Analysis stage to use
"the disk cache from Phase 11, rebuilding if missing," that claim would have been false
without a fix. Moved cycle-detection caching into `graph_analysis.py` as the single
shared owner (`get_cycle_analysis_cached()`), with `explainability.py` and the API's
`state.py` now importing it from there instead of duplicating it. The old cache file
(a bare `set`, pre-refactor format) is detected and transparently migrated to the new
format — no manual deletion needed. Verified in isolation before touching the pipeline:
first call detected the old format and rebuilt (151.1s, 10,651 cycles), second call
loaded instantly (0.00s) with identical results.

## Test 1: normal-mode run (nothing missing)

```
Step 1/6 EDA: SKIP (notebooks/eda_summary.md exists)
Step 2/6 Graph construction: SKIP (data/processed/transaction_graph.gpickle exists)
Step 3/6 Graph analysis: SKIP (report + cycle cache both exist)
Step 4/6 Feature engineering: SKIP (data/processed/account_features.csv exists)
Step 5a/6 Classical baselines: SKIP (models exist; use --retrain to force)
Step 5b/6 GraphSAGE: SKIP (model exists; use --retrain to force)
Step 6/6 Backend API: starting... → healthy in 17-22s
```

All six skip-checks fired correctly; the script's only real work was starting the
backend and polling `/health` until ready.

## Test 2: clean-checkout simulation (the main event)

Per the task, `data/processed/` (444MB: graph pickle, feature table, all four trained
models, cycle cache) was moved aside — not deleted — and the pipeline re-run from that
state. `data/raw/` and `notebooks/` were left untouched (matching the instruction's own
scope: "simulate this by temporarily renaming/moving `data/processed/`").

**Result: fully successful, unattended, ~26 minutes.**

```
22:07:09  Starting pipeline
22:07:09  Step 1/6 EDA: SKIP (notebooks/eda_summary.md exists — see note below)
22:07:09  Step 2/6 Graph construction: RUN         (30s)
22:07:39  Step 3/6 Graph analysis: RUN             (9m13s — includes full cycle rebuild)
22:16:52  Step 4/6 Feature engineering: RUN         (9m16s)
22:26:08  Step 5a/6 Classical baselines: RUN        (14s)
22:26:22  Step 5b/6 GraphSAGE: RUN                  (6m29s)
22:32:51  Step 6/6 Backend API: starting...
22:33:13  Backend healthy                            (22s)
```

No errors, no manual intervention, no crashes. Every `data/processed/` artifact was
regenerated with fresh timestamps confirming a genuine rebuild, not stale copies.

**Note on Step 1:** EDA was correctly skipped per its own check
(`notebooks/eda_summary.md`), since the task scoped this simulation to
`data/processed/` only — EDA produces no `data/processed/` output, so this is expected,
correct behavior for the test as specified, not a gap. A test that also wiped
`notebooks/` would exercise EDA's rebuild path too, but that wasn't what was asked.

## Test 3: full smoke test (both states)

Ran on **both** the freshly-rebuilt system and the restored-original system:

- All 5 endpoints (`/health`, `/accounts/flagged`, `/accounts/{id}/risk-score`,
  `/accounts/{id}/subgraph`, `/accounts/{id}/explanation`) returned correct data.
- 404 handling confirmed for a nonexistent account.
- Frontend loaded real flagged accounts from the live backend (not mock data) at every
  test point.
- Clicked into `8000E1590`: risk score, key features, GNNExplainer output, and the
  interactive Cytoscape sub-graph (23 nodes, 83 edges) all rendered correctly and
  matched the API's own responses exactly.

## Key finding: fixed seeds do not mean bit-reproducible rebuilds

This is the most important thing this test surfaced, and it would have been dishonest
to bury it. Comparing the clean-checkout rebuild's outputs to the values reported
throughout earlier phases:

| Component | Reproducibility | Evidence |
|---|---|---|
| Betweenness centrality, Louvain communities | **Exact** | Same "4 of top 20 illicit" result, same community stats |
| XGBoost (`n_jobs=-1`) | **Nearly exact, tiny drift** | AUC-ROC 0.8650 → 0.8658 |
| Cycle detection, giant-SCC portion | **Can vary substantially** | This run found only 5 cycles in that category vs. ~15-220 in earlier runs — because the search is a **120s wall-clock budget**, not an iteration count, so results depend on system load at execution time, not just the random seed |
| GraphSAGE (PyTorch, CPU) | **Meaningfully different** | AUC-ROC 0.9218 → 0.9239, AUC-PR 0.2740 → 0.2851, F2-optimal threshold 0.8727 → 0.8209 |

The GraphSAGE and XGBoost drift, despite `random_state=42` / `torch.manual_seed(42)`
being set everywhere, is almost certainly floating-point non-associativity in
multi-threaded CPU matrix/histogram operations — a well-known limitation, not a bug in
this project's code, but one that's easy to assume away until an actual rebuild proves
otherwise.

**A concrete consequence this test caught and fixed:** because the backup/restore for
this test was scoped to `data/processed/` only (per the task), restoring the original
model files left `notebooks/gnn_baseline_summary.md` describing the *freshly retrained*
model (0.9239/0.2851/0.8209) rather than the *restored* one now actually being served
(0.9218/0.2740/0.8727) — a real, live inconsistency between documentation and the
running system. Fixed by loading the restored `graphsage_baseline.pt` and re-running
evaluation only (no retraining); this reproduced the original numbers exactly,
confirming the restored model file is bit-identical to the original, and the notebook
was rewritten to match. The other regenerated notebooks
(`graph_analysis_summary.md`, `feature_engineering_summary.md`,
`baseline_model_summary.md`) were left as their honest post-rebuild versions rather
than reverted, since they're standalone reports (nothing live serves them the way the
API serves the GNN model) and reverting them would misrepresent what a real rebuild
actually produces.

**Recommendation, not fixed in this phase (out of scope):** the frontend header
hardcodes "AUC-ROC 0.922 · AUC-PR 0.274" as static text rather than fetching it from
the API. After any retrain, this display can silently go stale relative to the model
actually being served (as it briefly did during this test). Worth wiring to a live
endpoint in a future pass.

## Restoration verification

`data/processed/` was restored (not deleted, moved back from the backup) after both
test-mode runs completed. Verified correct via a **before/after threshold check**:
health check showed `threshold: 0.8208966...` on the fresh rebuild, and
`threshold: 0.8726518...` (the original) after restoring — confirmed twice, once via
direct `curl` and once visually in the frontend header/subtitle, which were consistent
with each other and with every earlier phase's reported figures.

## Honest accounting: what is NOT automated

- **Downloading the raw Kaggle dataset.** `data/raw/HI-Small_Trans.csv` and
  `HI-Small_Patterns.txt` must be manually downloaded from Kaggle (requires
  authentication) before this script can do anything. The script fails fast with an
  explicit message rather than attempting to fetch them.
- **Python environment setup.** Creating the venv and `pip install -r requirements.txt`
  is not automated — the script checks for `venv/bin/activate` and fails fast with the
  exact command to run if it's missing, rather than silently creating an environment.
- **Frontend setup and startup.** `npm install` and `npm run dev` are entirely separate
  from `run_pipeline.sh`, by design — the task scoped the script to end at "starting the
  backend API," and frontend startup was tested manually in this session, in a second
  terminal, per the existing README.
- **`explainability.py` (Phase 10) is not part of the automated pipeline.** It generates
  demo PNGs and a written report for a handful of sample accounts — genuinely useful
  output, but not a prerequisite for the API (which gets its cycle cache from the Graph
  Analysis stage, not from explainability.py). Running it is still a manual step if you
  want fresh explainability artifacts.
- **Stopping the servers.** No stop script exists; this session did it manually via the
  PID file the pipeline script writes (`scripts/.run/backend.pid`) and the frontend's
  process ID (which isn't tracked anywhere since the pipeline script never starts the
  frontend).
- **CORS origin is hardcoded** to `http://localhost:5173` in `src/api/main.py` — would
  need a manual edit if the frontend's dev port ever changes.

## Files changed in this phase

- New: `scripts/run_pipeline.sh`
- `src/graph_analysis.py`: added `compute_cycle_analysis()` / `get_cycle_analysis_cached()`
  (moved/generalized from `explainability.py`), `main()` now uses the cache
- `src/explainability.py`: `compute_all_cycles()`/local cache logic removed;
  `get_cycles_cached()` is now a thin wrapper around the shared function in
  `graph_analysis.py`
- `notebooks/gnn_baseline_summary.md`: regenerated via eval-only pass to match the
  restored model (see Key Finding above)
- `data/processed/all_cycles_cache.pkl`: migrated from the old bare-`set` format to the
  new dict format (transparently, no manual step)
