# Financial Crime Graph Analytics

Explainable graph learning for anti-money laundering (AML) on the IBM AML synthetic
transaction dataset (HI-Small). The project models 5,078,345 transactions between 515,080
accounts as a directed transaction graph, analyses its structure, and compares a GraphSAGE
graph neural network against XGBoost and Random Forest baselines. Each flagged account can be
explained with GNNExplainer, and everything is served through a FastAPI backend and a
React/Cytoscape.js analyst dashboard.

Pipeline: transaction graph → graph analysis (centrality / communities / cycles) → engineered
feature table → classical (XGBoost / Random Forest) and GraphSAGE models → GNNExplainer
explainability → FastAPI backend → React/Cytoscape.js dashboard.

## Results

All models share one stratified 70/10/20 train/validation/test split (`src/splits.py`), use
leakage-free features, and are evaluated at an F2-optimal threshold chosen on the validation
set. Test set: 103,016 accounts, of which 1,271 are illicit.

| Model         | AUC-ROC | AUC-PR | Precision | Recall | F1     |
|---------------|---------|--------|-----------|--------|--------|
| XGBoost       | 0.8534  | 0.1229 | 0.0918    | 0.3698 | 0.1470 |
| Random Forest | 0.8470  | 0.0864 | 0.0673    | 0.4186 | 0.1159 |
| **GraphSAGE** | **0.9179** | **0.2470** | **0.1639** | **0.4784** | **0.2442** |

GraphSAGE doubles the AUC-PR of the best tree model and, at its operating point, finds 138
more illicit accounts than XGBoost while raising 1,551 fewer false alerts.

## Repository Structure

- `data/raw/`: original source files (HI-Small_Trans.csv, HI-Small_Patterns.txt). Not in the
  repository; download them as described below.
- `data/processed/`: built graph, feature table, trained GraphSAGE model, metrics and cycle
  cache (large files are stored with Git LFS).
- `notebooks/`: per-phase summary reports (EDA, graph analysis, feature engineering, model
  evaluation, explainability, API) and explainability figures.
- `src/`: pipeline scripts (`eda.py`, `build_graph.py`, `graph_analysis.py`, `splits.py`,
  `feature_engineering.py`, `train_baseline.py`, `build_pyg_data.py`, `train_gnn.py`,
  `explainability.py`).
- `src/api/`: FastAPI backend (see `notebooks/api_summary.md`).
- `src/frontend/`: React + TypeScript + Vite + Cytoscape.js dashboard.
- `scripts/run_pipeline.sh`: end-to-end pipeline runner, from raw data to a running backend.

## Requirements

- Python 3.12
- Node.js 20 or newer (with npm)
- Git LFS (needed to download the large files in `data/processed/`)
- About 8 GB of free RAM (the in-memory graph takes roughly 1.5 to 2.5 GB)

## Setup

**1. Clone the repository (with Git LFS)**

```bash
git lfs install
git clone https://github.com/Varunkumar-07/Financial-crime-graph-analytics.git
cd Financial-crime-graph-analytics
```

If you cloned before installing Git LFS, run `git lfs pull` inside the repository.

**2. Create the Python environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Install the frontend dependencies**

```bash
cd src/frontend
npm install
cd ../..
```

**4. Download the dataset (only needed to rebuild the pipeline)**

The trained model and processed data are already in `data/processed/`, so the demo runs
without the raw data. To rebuild everything from scratch, download the IBM AML dataset from
Kaggle:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Place `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/raw/`.

## Running the Full Stack Locally

Two servers, run in separate terminals from the project root.

**1. Backend (FastAPI, port 8000)**

```bash
source venv/bin/activate
cd src
uvicorn api.main:app --reload --app-dir .
```

Wait for `[startup] Ready in ...s` in the logs. Startup takes about 7 s because the
cycle-detection cache is included at `data/processed/all_cycles_cache.pkl`; if that file is
missing, the first run rebuilds it (about 150 to 300 s, one time only). Verify with
`curl http://127.0.0.1:8000/health`. Interactive API docs: http://127.0.0.1:8000/docs

**2. Frontend (Vite dev server, port 5173)**

```bash
cd src/frontend
npm run dev
```

Open http://localhost:5173. The dashboard loads real flagged accounts from the backend (not
mock data), so the backend must already be running. CORS is configured for `localhost:5173`
only (see `src/api/main.py`).

Click any account to see its risk score, engineered features, GNNExplainer attribution, and
an interactive Cytoscape.js subgraph (zoom, pan, and click to move between accounts). Case
status (flagged / reviewing / cleared / escalated) is client-side only and resets on page
reload.

## Rebuilding the Pipeline

With the venv set up and the raw data in `data/raw/`, run from the project root:

```bash
scripts/run_pipeline.sh
```

The script runs EDA, graph construction, graph analysis, feature engineering, model training
and finally starts the backend. Each stage is skipped if its output already exists, so a
re-run only starts the backend. To force the models to retrain:

```bash
scripts/run_pipeline.sh --retrain
```

A full rebuild from raw data takes about 26 minutes. The XGBoost and Random Forest model
files are not stored in the repository, so they are recreated by this step; the dashboard
only needs the GraphSAGE model, which is included. The frontend is not started by the script.

## API Endpoints

| Method | Path                         | Purpose                                                   |
|--------|------------------------------|-----------------------------------------------------------|
| GET    | `/health`                    | Service status, cycle-cache status and active threshold   |
| GET    | `/model/info`                | Model name and test-set AUC-ROC / AUC-PR, computed live   |
| GET    | `/accounts/flagged`          | Paginated, server-sorted list of flagged accounts         |
| GET    | `/accounts/{id}/risk-score`  | Predicted probability, label and key features             |
| GET    | `/accounts/{id}/subgraph`    | 1 to 2 hop subgraph for visualisation                     |
| GET    | `/accounts/{id}/explanation` | GNNExplainer feature/edge attributions and cycles         |

## Team

Presidency School of Artificial Intelligence & Advanced Computing, Presidency University,
Bengaluru (B.Tech CSE (AI&ML), 2026-27).

- Varun Kumar (20231CAI0152)
- Mohammed Bilal Sharief (20231CAI0145)
- Parshwa Dhanapal Bagi (20231CAI0150)
- MK Reddy Venkata Santhosh (20231CAI0154)

**Guide:** Ms. Surya C, Assistant Professor, PSAIAC

## Dataset Citation

E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel, and K. Atasu,
"Realistic Synthetic Financial Transactions for Anti-Money Laundering Models," NeurIPS
Datasets and Benchmarks Track, 2023.
