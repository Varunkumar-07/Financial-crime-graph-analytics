#!/usr/bin/env bash
#
# End-to-end pipeline: raw data -> transaction graph -> graph analysis
# -> feature table -> trained models -> running backend API.
#
# Idempotent by default: each stage is skipped if its expected output
# file(s) already exist, so re-running this script after a successful
# run just starts the backend in a few seconds. Pass --retrain to force
# the model-training stages (classical baselines + GraphSAGE) to rerun
# even if trained models already exist.
#
# NOT automated by this script (see notebooks/integration_test_summary.md
# for the full list):
#   - Downloading the raw Kaggle dataset into data/raw/ (requires Kaggle
#     authentication) — this script fails fast with a clear message if
#     data/raw/HI-Small_Trans.csv is missing, rather than trying to fetch it.
#   - Creating the Python venv / installing requirements.txt.
#   - Installing frontend npm dependencies / starting the frontend dev
#     server (out of this script's scope — see README.md).
#
# Usage:
#   scripts/run_pipeline.sh              # skip stages whose output exists
#   scripts/run_pipeline.sh --retrain    # force model-training stages to rerun
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RETRAIN=false
for arg in "$@"; do
  case "$arg" in
    --retrain) RETRAIN=true ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (use --retrain or --help)" >&2
      exit 1
      ;;
  esac
done

RUN_DIR="$PROJECT_ROOT/scripts/.run"
mkdir -p "$RUN_DIR"
BACKEND_LOG="$RUN_DIR/backend.log"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"

log() { echo "[pipeline $(date '+%H:%M:%S')] $*"; }

# --- Prerequisite checks (fail fast with an actionable message rather
# than a confusing downstream Python traceback) ---
if [ ! -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  echo "ERROR: venv not found at $PROJECT_ROOT/venv" >&2
  echo "  Manual step required: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
source "$PROJECT_ROOT/venv/bin/activate"

if [ ! -f "$PROJECT_ROOT/data/raw/HI-Small_Trans.csv" ]; then
  echo "ERROR: data/raw/HI-Small_Trans.csv not found." >&2
  echo "  Manual step required: download the IBM AML synthetic dataset (HI-Small) from Kaggle" >&2
  echo "  (requires Kaggle account/API auth) and place HI-Small_Trans.csv + HI-Small_Patterns.txt in data/raw/." >&2
  exit 1
fi

log "Starting pipeline (retrain=$RETRAIN)"

# --- Step 1: EDA ---
# EDA writes only a report (no data/processed artifact), so its skip-check
# is the report file itself.
if [ -f "notebooks/eda_summary.md" ]; then
  log "Step 1/6 EDA: SKIP (notebooks/eda_summary.md exists)"
else
  log "Step 1/6 EDA: RUN"
  python src/eda.py
fi

# --- Step 2: Graph construction ---
if [ -f "data/processed/transaction_graph.gpickle" ]; then
  log "Step 2/6 Graph construction: SKIP (data/processed/transaction_graph.gpickle exists)"
else
  log "Step 2/6 Graph construction: RUN"
  python src/build_graph.py
fi

# --- Step 3: Graph analysis (centrality, community, cycles) ---
# Skips only if BOTH the report and the cycle-detection disk cache exist —
# checking the report alone would be wrong: if data/processed/ was wiped
# but notebooks/ wasn't, the report would still be there while the cache
# (and everything downstream needs) would be gone.
if [ -f "notebooks/graph_analysis_summary.md" ] && [ -f "data/processed/all_cycles_cache.pkl" ]; then
  log "Step 3/6 Graph analysis: SKIP (report + cycle cache both exist)"
else
  log "Step 3/6 Graph analysis: RUN (betweenness/Louvain always recomputed here; cycle detection uses/rebuilds the disk cache internally)"
  python src/graph_analysis.py
fi

# --- Step 4: Feature engineering ---
if [ -f "data/processed/account_features.csv" ]; then
  log "Step 4/6 Feature engineering: SKIP (data/processed/account_features.csv exists)"
else
  log "Step 4/6 Feature engineering: RUN"
  python src/feature_engineering.py
fi

# --- Step 5: Model training ---
CLASSICAL_DONE=false
if [ -f "data/processed/xgboost_baseline.json" ] && [ -f "data/processed/random_forest_baseline.pkl" ]; then
  CLASSICAL_DONE=true
fi
if [ "$CLASSICAL_DONE" = true ] && [ "$RETRAIN" = false ]; then
  log "Step 5a/6 Classical baselines (XGBoost/RF): SKIP (models exist; use --retrain to force)"
else
  log "Step 5a/6 Classical baselines (XGBoost/RF): RUN"
  python src/train_baseline.py
fi

GNN_DONE=false
if [ -f "data/processed/pyg_data.pt" ] && [ -f "data/processed/graphsage_baseline.pt" ]; then
  GNN_DONE=true
fi
if [ "$GNN_DONE" = true ] && [ "$RETRAIN" = false ]; then
  log "Step 5b/6 GraphSAGE: SKIP (model exists; use --retrain to force)"
else
  log "Step 5b/6 GraphSAGE: RUN"
  python src/build_pyg_data.py
  python src/train_gnn.py
fi

# --- Step 6: Start backend API ---
if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  log "Step 6/6 Backend API: already running (PID $(cat "$BACKEND_PID_FILE"))"
else
  log "Step 6/6 Backend API: starting..."
  (
    cd "$PROJECT_ROOT/src"
    nohup uvicorn api.main:app --host 127.0.0.1 --port 8000 > "$BACKEND_LOG" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"
  )

  log "Waiting for backend health check (up to 180s)..."
  READY=false
  for _ in $(seq 1 90); do
    if curl -s -f http://127.0.0.1:8000/health > /dev/null 2>&1; then
      READY=true
      break
    fi
    sleep 2
  done

  if [ "$READY" = true ]; then
    log "Backend healthy: http://127.0.0.1:8000/docs (PID $(cat "$BACKEND_PID_FILE"), logs: $BACKEND_LOG)"
  else
    log "ERROR: backend did not become healthy within 180s. Check $BACKEND_LOG"
    kill "$(cat "$BACKEND_PID_FILE")" 2>/dev/null || true
    rm -f "$BACKEND_PID_FILE"
    exit 1
  fi
fi

log "Pipeline complete. Frontend is NOT started by this script — see README.md (cd src/frontend && npm install && npm run dev)."
