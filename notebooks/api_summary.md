# Backend API Summary

FastAPI service exposing the trained GraphSAGE model (test AUC-ROC 0.922, AUC-PR 0.274),
the transaction graph, and GNNExplainer-based explanations over HTTP.

## Running it

```bash
cd src
uvicorn api.main:app --reload --app-dir .
```

Interactive docs at `http://127.0.0.1:8000/docs` (Swagger UI) — confirmed rendering correctly
with all 5 endpoints grouped under `health` and `accounts` tags.

## Structure

```
src/api/
├── main.py                 # App instance, lifespan startup, router registration
├── models/
│   ├── state.py             # AppState singleton — loads model/graph/features/cycles ONCE
│   └── inference.py         # Re-exports subgraph/GNNExplainer helpers from explainability.py
├── routers/
│   ├── health.py             # GET /health
│   └── accounts.py           # GET /accounts/{flagged,{id}/risk-score,{id}/subgraph,{id}/explanation}
└── schemas/
    ├── account.py, subgraph.py, explanation.py, health.py   # Pydantic request/response models
```

## Startup performance: the cycle-detection cache

Phase 5/10's cycle detection (2-6 length directed cycles) costs ~140-300s because of a
120s-capped partial sample over the graph's one giant 17,075-node strongly-connected
component (see `graph_analysis.py` for the full methodology and caveat — that cap still
applies here; the giant-SCC portion of the cycle set is a lower bound, not exhaustive).
Recomputing that on every API restart would make local development painful, so
`explainability.get_cycles_cached()` was added: it pickles the result to
`data/processed/all_cycles_cache.pkl` on first computation and loads from that file on
every subsequent startup.

Confirmed effect: total app startup time dropped from ~150-300s to **6.8s** once the cache
existed. If that cache file is ever deleted, the next startup will silently take the full
~150-300s again — this is expected, not a bug.

## Model/graph loading (once, at startup — confirmed)

`AppState.load()` runs inside the FastAPI `lifespan` context manager, called exactly once
when the server starts, and every route handler reads from the already-populated `state`
singleton — no per-request re-loading. Confirmed via the startup log (each `[startup] ...`
line prints exactly once) and via response latency (`/accounts/{id}/risk-score` and
`/subgraph` are sub-100ms; only `/explanation`, which runs GNNExplainer's own optimization
loop per request by design, takes ~0.3-3s depending on the account's 3-hop neighborhood size).

At startup, in addition to the model and graph, the full 515,080-account probability array
is computed once (`proba_all`) and the flagged-accounts list is pre-sorted once — so
`/accounts/flagged` pagination is just an array slice, not a per-request sort.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, accounts loaded, cycles cached, active threshold |
| GET | `/accounts/flagged?page=&page_size=` | Paginated accounts with probability ≥ threshold (0.8727), sorted descending |
| GET | `/accounts/{account_id}/risk-score` | Predicted probability, label, and key engineered features |
| GET | `/accounts/{account_id}/subgraph` | 1-2 hop ego-subgraph (nodes + edges) as JSON, ready for a frontend graph viz |
| GET | `/accounts/{account_id}/explanation` | GNNExplainer top features/edges + cycle membership for that account |

All `/accounts/{account_id}/...` routes return **404** with a clear message
(`Account 'X' not found in graph`) for an unknown account_id, rather than a 500 or silent
wrong answer — validated with a live test against a nonexistent ID.

### Important scope note on `/accounts/flagged`

This endpoint scores **all 515,080 accounts** in the graph (train + val + test combined),
not just the held-out test set — this mirrors how the model would actually be used in
production (scoring the whole population, not a labeled evaluation split). As a result,
**12,748 accounts (2.47%) are currently flagged** — do not confuse this figure with the
test-set confusion matrix from Phase 8/9 (505 TP / 2,064 FP out of 103,016 test accounts),
which is the correct source for measuring true model accuracy. This endpoint's numbers are
inflated by accounts the model saw during training and will look better than true
generalization performance.

## Verified example responses (live test, not just described)

**GET /health**
```json
{"status": "ok", "model_loaded": true, "n_accounts": 515080, "n_cycles_cached": 10650, "threshold": 0.8726518750190735}
```

**GET /accounts/8000E1590/risk-score** (Phase 10's "confident true positive" case — probability matches exactly)
```json
{"account_id": "8000E1590", "probability": 0.9824122786521912, "predicted_label": "illicit",
 "threshold": 0.8726518750190735, "true_label": "illicit",
 "key_features": {"in_degree": 6.0, "out_degree": 21.0, "cycle_count": 30.0, "community_illicit_density": 0.0229, "...": "..."}}
```

**GET /accounts/80C7FA5E0/subgraph** — returned 7 nodes / 16 edges, matching the Phase 10 visualization
node-for-node, including the single illicit edge from hub account `100428660` into `80C7F9990`.

**GET /accounts/806CD7380/explanation** (Phase 10's false-negative case) — `cycle_membership_count: 8`,
`cycle_lengths: [2,2,3,3,3,3,3,3]`, matching Phase 10 exactly. Top-edge attribution values differ
slightly run-to-run since GNNExplainer's mask optimization isn't seeded — expected variance in the
explanation method itself, not an API bug.

**GET /accounts/flagged?page=1&page_size=5** — returned top 5 by probability (0.99999+ range),
`total_flagged: 12748`, `total_pages: 2550`. Page 2 confirmed to return different, non-overlapping accounts.

**GET /accounts/DOES_NOT_EXIST/risk-score** → `404 {"detail": "Account 'DOES_NOT_EXIST' not found in graph"}`

**GET /accounts/flagged?page_size=99999** → `422` (Pydantic's built-in `le=200` constraint rejects it
before it reaches the handler) — confirms query-parameter validation, not just path-parameter validation.

## Known limitations / not addressed in this phase

- No auth/rate-limiting (out of scope for this phase — internal/dev API only).
- `/explanation` is synchronous and CPU-bound; a production deployment serving concurrent
  explanation requests would want a task queue rather than blocking the request thread.
- `true_label` is exposed in `/risk-score` for demo/evaluation convenience against this
  labeled dataset — a real production deployment would never have this at inference time.
