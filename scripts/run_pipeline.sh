#!/usr/bin/env bash
#
# Runs the full pipeline: raw data -> graph -> analysis -> features -> models -> API.
# Steps whose output already exists are skipped. Use --retrain to force model training.
#
# Not handled here: downloading the Kaggle data, setting up the venv, and the frontend.
#
# Usage:
#   scripts/run_pipeline.sh            # skip finished steps
#   scripts/run_pipeline.sh --retrain  # retrain models
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
      sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

# Check prerequisites
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

# Set when graph, cycles or features are rebuilt, so later steps rerun too.
UPSTREAM_CHANGED=false

# Step 1: EDA (only writes a report, so check for that)
if [ -f "notebooks/eda_summary.md" ]; then
  log "Step 1/7 EDA: SKIP (notebooks/eda_summary.md exists)"
else
  log "Step 1/7 EDA: RUN"
  python src/eda.py
fi

# Step 2: Build the graph
if [ -f "data/processed/transaction_graph.gpickle" ]; then
  log "Step 2/7 Graph construction: SKIP (data/processed/transaction_graph.gpickle exists)"
else
  log "Step 2/7 Graph construction: RUN"
  python src/build_graph.py
  UPSTREAM_CHANGED=true
  # The old cycle cache belongs to the old graph.
  rm -f data/processed/all_cycles_cache.pkl
fi

# Step 3: Graph analysis (skip only if both report and cycle cache exist)
if [ -f "notebooks/graph_analysis_summary.md" ] && [ -f "data/processed/all_cycles_cache.pkl" ]; then
  log "Step 3/7 Graph analysis: SKIP (report + cycle cache both exist)"
else
  log "Step 3/7 Graph analysis: RUN"
  [ -f "data/processed/all_cycles_cache.pkl" ] || UPSTREAM_CHANGED=true
  python src/graph_analysis.py
fi

# Step 4: Feature engineering
if [ -f "data/processed/account_features.csv" ] && [ "$UPSTREAM_CHANGED" = false ]; then
  log "Step 4/7 Feature engineering: SKIP (data/processed/account_features.csv exists)"
else
  log "Step 4/7 Feature engineering: RUN"
  python src/feature_engineering.py
  UPSTREAM_CHANGED=true
fi

# Step 5: Model training (reruns on --retrain or when features changed)
if [ -f "data/processed/xgboost_baseline.json" ] && [ -f "data/processed/random_forest_baseline.pkl" ] \
   && [ "$RETRAIN" = false ] && [ "$UPSTREAM_CHANGED" = false ]; then
  log "Step 5a/7 Classical baselines (XGBoost/RF): SKIP (models exist; use --retrain to force)"
else
  log "Step 5a/7 Classical baselines (XGBoost/RF): RUN"
  python src/train_baseline.py
fi

GNN_RETRAINED=false
if [ -f "data/processed/pyg_data.pt" ] && [ -f "data/processed/graphsage_baseline.pt" ] \
   && [ "$RETRAIN" = false ] && [ "$UPSTREAM_CHANGED" = false ]; then
  log "Step 5b/7 GraphSAGE: SKIP (model exists; use --retrain to force)"
else
  log "Step 5b/7 GraphSAGE: RUN"
  if [ ! -f "data/processed/pyg_data.pt" ] || [ "$UPSTREAM_CHANGED" = true ]; then
    python src/build_pyg_data.py
  fi
  python src/train_gnn.py
  GNN_RETRAINED=true
fi

# Step 6: Explainability case studies (rerun when the GNN changes)
if [ -f "notebooks/explainability_cases.json" ] && [ "$GNN_RETRAINED" = false ]; then
  log "Step 6/7 Explainability: SKIP (notebooks/explainability_cases.json exists)"
else
  log "Step 6/7 Explainability: RUN"
  python src/explainability.py
fi

# Step 7: Start the backend API
if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  log "Step 7/7 Backend API: already running (PID $(cat "$BACKEND_PID_FILE"))"
else
  log "Step 7/7 Backend API: starting..."
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
