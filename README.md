# Financial Crime Graph Analytics

Graph analytics project on the IBM AML synthetic transaction dataset (Kaggle: HI-Small):
transaction graph → graph analysis (centrality/communities/cycles) → engineered feature
table → classical (XGBoost/Random Forest) and GraphSAGE GNN baselines → GNNExplainer
explainability → FastAPI backend → React/Cytoscape.js dashboard.

## Structure

- `data/raw/` — original, unmodified source files (HI-Small_Trans.csv, HI-Small_Patterns.txt)
- `data/processed/` — built graph, feature table, trained models, cycle cache
- `notebooks/` — per-phase summaries (EDA, graph analysis, feature engineering, model
  evaluation, explainability, API docs)
- `src/` — pipeline scripts (`eda.py`, `build_graph.py`, `graph_analysis.py`,
  `feature_engineering.py`, `train_baseline.py`, `train_gnn.py`, `explainability.py`)
- `src/api/` — FastAPI backend (see `notebooks/api_summary.md`)
- `src/frontend/` — React + Vite + Cytoscape.js dashboard

## Status

All phases through the frontend dashboard are complete. GraphSAGE (test AUC-ROC 0.922,
AUC-PR 0.274) outperforms the XGBoost baseline (AUC-ROC 0.865, AUC-PR 0.146). Docker/
deployment has not been started.

## Running the full stack locally

Two servers, run in separate terminals.

**1. Backend (FastAPI, port 8000)**

```bash
cd "financial-crime-graph"
source venv/bin/activate
cd src
uvicorn api.main:app --reload --app-dir .
```

Wait for `[startup] Ready in ...s` in the logs (first run builds the cycle-detection cache,
~150-300s one-time; every run after that is ~7s since it's cached at
`data/processed/all_cycles_cache.pkl`). Verify with `curl http://127.0.0.1:8000/health`.
Interactive API docs: http://127.0.0.1:8000/docs

**2. Frontend (Vite dev server, port 5173)**

```bash
cd "financial-crime-graph/src/frontend"
npm install   # first time only
npm run dev
```

Open http://localhost:5173 — the dashboard loads real flagged accounts from the backend
(not mock data). The backend must already be running; CORS is configured for
`localhost:5173` only (see `src/api/main.py`).

Click any account to see its risk score, engineered features, GNNExplainer attribution,
and an interactive Cytoscape.js sub-graph (zoom/pan/click-to-navigate between accounts).
Case status (flagged/reviewing/cleared/escalated) is client-side only and resets on page
reload — no backend persistence yet.
