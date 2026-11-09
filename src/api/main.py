"""
FastAPI app: risk scores, subgraphs and explanations per account.
Everything is loaded once at startup (see api/models/state.py).

Usage: uvicorn api.main:app --reload --app-dir src
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.models.state import state
from api.routers import accounts, health, model


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

# Allowed browser origins (Vite dev server by default). Override with
# the CORS_ALLOWED_ORIGINS env var.
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
