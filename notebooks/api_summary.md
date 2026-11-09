# Backend API Summary

FastAPI service exposing the trained GraphSAGE model (test AUC-ROC 0.914, AUC-PR 0.232),
the transaction graph, and GNNExplainer-based explanations over HTTP.

## Running it

```bash
cd src
uvicorn api.main:app --reload --app-dir .
```

Interactive docs at `http://127.0.0.1:8000/docs` (Swagger UI).

## Structure

```
src/api/
├── main.py                 # App instance, lifespan startup, CORS, router registration
├── models/
│   ├── state.py             # AppState singleton — loads model/graph/features/cycles ONCE
│   └── inference.py         # Re-exports subgraph/GNNExplainer helpers from explainability.py
├── routers/
│   ├── health.py             # GET /health
│   ├── model.py              # GET /model/info
│   └── accounts.py           # GET /accounts/{flagged,{id}/risk-score,{id}/subgraph,{id}/explanation}
└── schemas/                  # Pydantic response models
```

## Startup

`AppState.load()` runs once in the FastAPI `lifespan`; every route reads from the loaded
`state`. At startup it also scores all 515,080 accounts once (`proba_all`), picks the F2
threshold on the validation set, computes the live test metrics served by `/model/info`, and
builds the flagged-accounts table.

Cycles come from `data/processed/all_cycles_cache.pkl` (startup ~7 s): 5,155 exact cycles of
length 2-6. If the file is missing, the first startup rebuilds it (a few seconds).

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, accounts loaded, cycles cached, active threshold |
| GET | `/model/info` | Model name, live test AUC-ROC / AUC-PR, threshold |
| GET | `/accounts/flagged?page=&page_size=&sort_by=&sort_order=&search=` | Flagged accounts (probability ≥ threshold), filtered by `search` (account ID substring) and sorted on the full set per request (stable sort), then paged |
| GET | `/accounts/{account_id}/risk-score` | Probability, label and key features (`in_out_amount_ratio` is `null` if the account never sent money) |
| GET | `/accounts/{account_id}/subgraph` | 1-2 hop subgraph; for large accounts only the 150 busiest neighbours (`truncated: true`) |
| GET | `/accounts/{account_id}/explanation` | GNNExplainer top features and account pairs, plus cycle membership; `422` if the 3-hop neighbourhood exceeds 30,000 accounts |

- Unknown accounts return **404** (`Account 'X' not found in graph`).
- `page_size` above 200 returns **422**.
- Explanations are cached (256 accounts) and run one at a time, since GNNExplainer reseeds
  torch globally.
- Edge importance is per account pair (the GNN passes messages both ways), so each pair is
  listed once and the frontend highlights it in whichever direction the money moved.

### Scope note on `/accounts/flagged`

This endpoint scores **all 515,080 accounts** (train + val + test), as the model would be
used in practice. **23,210 accounts (4.51%) are currently flagged**. Do not confuse this with
the test-set confusion matrix (645 TP / 3,953 FP out of 103,016 test accounts), which is the
correct measure of model accuracy; the flagged list includes accounts seen in training and
looks better than true generalization.

## Known limitations

- No auth or rate limiting (local demo API only).
- `/explanation` is CPU-bound; a real deployment would use a task queue.
- `true_label` is exposed for demo/evaluation against this labeled dataset; a real
  deployment would not have it at inference time.
