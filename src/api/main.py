"""
FastAPI app for the financial-crime-graph project: serves GraphSAGE
risk scores, local sub-graphs, and GNNExplainer-based explanations for
accounts in the transaction graph.

The model, graph, feature table, and cycle cache are all loaded ONCE
at startup via AppState.load() (see api/models/state.py) — not
per-request. Run with:

    uvicorn api.main:app --reload --app-dir src
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from api.models.state import state  # noqa: E402
from api.routers import accounts, health, model  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Financial Crime Graph Analytics API",
    description="Risk scores, sub-graphs, and explanations for accounts in the transaction graph, "
    "backed by a GraphSAGE model. See GET /model/info for this model's live "
    "eval metrics — not hardcoded here, since retraining changes them.",
    version="1.0.0",
    lifespan=lifespan,
)

# Origins that may call this API directly from a browser. Needed whenever a
# frontend calls this API cross-origin (local Vite dev server on a different
# port; a containerized frontend published on a different host port). When
# the frontend instead reaches the API via its own nginx reverse proxy
# (same-origin from the browser's perspective — see frontend/nginx.conf and
# docker-compose.yml), CORS isn't even invoked, but the setting is left
# permissive there too so the API remains directly callable for
# debugging/testing regardless of deployment shape.
# Configurable via env var — never hardcoded — because the right origin
# list depends on deployment (local dev vs. docker-compose vs. a future
# live host), and hardcoding it here would need a code change every time.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(accounts.router)
app.include_router(model.router)
