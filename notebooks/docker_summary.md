# Docker & Render-Readiness Summary

Two parts: containerizing the stack (Dockerfiles, compose, the seeded-dataset
decision), and making those images actually deployable to Render (dynamic ports,
runtime-configurable backend URL, git/size considerations). The second part exists
because Render's deployment model — separately-deployed services, ports assigned
dynamically, env vars set post-deploy via dashboard — breaks a few assumptions that
are fine for local docker-compose but wrong for a real host.

## Images

- **`Dockerfile.backend`** — multi-stage. Builder stage installs Python packages
  into an isolated `/install` prefix (including a **CPU-only torch wheel** from
  `download.pytorch.org/whl/cpu` — the default PyPI Linux wheel bundles CUDA
  runtime libraries this project never uses, since GraphSAGE runs on CPU; see
  `gnn_baseline_summary.md`). Runtime stage copies only that prefix, the app code,
  and the seeded dataset — no pip cache, no build toolchain. Final size: **2.39GB**
  (1.34GB of that is the ML stack — torch/torch-geometric/scikit-learn/pandas —
  the rest is the Debian/Python base plus the bundled dataset).
- **`Dockerfile.frontend`** — multi-stage. Builder stage runs `npm ci && npm run
  build` with Node; runtime stage is nginx + the static bundle only, no Node
  toolchain carried over. Final size: **77MB**.
- **`requirements-api.txt`** — a minimal subset of `requirements.txt` for the
  backend image. Excludes `xgboost` (used only by `src/train_baseline.py`, never
  imported by the running API) — verified empirically by simulating its absence
  from `sys.modules` and confirming `api.main` still imports cleanly, not just by
  inspection.

## The seeded-dataset decision

**Baked directly into the backend image**, not a volume mount or startup download.
Verified exactly which files `api/models/state.py` loads at runtime — 5 files,
~384MB total (`transaction_graph.gpickle` 270MB, `pyg_data.pt` 64MB,
`account_features.csv` 50MB, `graphsage_baseline.pt` 44KB,
`all_cycles_cache.pkl` 385KB). `data/raw/` (454MB of raw CSVs) and the unused
classical-baseline files (`xgboost_baseline.json`, `random_forest_baseline.pkl`,
~60MB combined) are excluded — the API never touches them.

Rationale: satisfies "always has data to show" literally with zero external
dependencies, no volume-mount requirement a cloud host might not support easily,
and avoids the cycle-cache's ~150-300s cold-rebuild cost that would otherwise hit
every fresh container start.

## docker-compose.yml (local dev/testing)

Two services, `backend` (port 8000 published) and `frontend` (port 3000→80,
depends on backend's healthcheck). Frontend's nginx reaches backend via Docker
Compose's embedded DNS (`http://backend:8000`) — that hostname only resolves
inside this compose network, which is exactly why it can't be hardcoded for Render
(see below).

## Render-readiness fixes (this phase)

Three problems existed that only matter once backend and frontend are deployed as
**separate** Render services rather than sharing one Docker Compose network:

**1. Hardcoded port.** `CMD ["uvicorn", ..., "--port", "8000"]` was exec-form JSON
— no shell, no env var expansion possible. Render assigns the actual port
dynamically via `$PORT` (commonly 10000, never assume a fixed value) and routes
traffic to whatever the container binds to; a hardcoded port would silently break.
Fixed: shell-form `CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}`, with
`ENV PORT=8000` as a default that only matters for docker-compose (nothing sets
`$PORT` there). Verified by running the container with `-e PORT=9999` and
confirming uvicorn's own startup logs appeared (not a shell syntax error) before
stopping it — didn't wait for the known OOM (see below) since that's an unrelated,
already-understood issue.

**2. Hardcoded `proxy_pass http://backend:8000/`.** `backend` only resolves inside
the Compose network; on Render, the backend has its own public URL that doesn't
exist until *after* the backend service deploys. Fixed by converting
`nginx.conf` → `nginx.conf.template`, using nginx's official image's built-in
behavior: any `*.template` file under `/etc/nginx/templates/` gets `envsubst`-processed
into `/etc/nginx/conf.d/` **at container start**, not build time — so
`proxy_pass ${BACKEND_URL};` and `listen ${PORT};` resolve from real environment
variables set on the running container, settable via Render's dashboard after the
backend's URL is known, without rebuilding the image. Verified end-to-end: ran the
frontend image standalone with `-e BACKEND_URL=http://example-backend.onrender.com
-e PORT=8080`, confirmed the entrypoint log line ("Running envsubst on
.../default.conf.template"), inspected the rendered `/etc/nginx/conf.d/default.conf`
and confirmed both substitutions landed correctly while nginx's own runtime
variables (`$host`, `$remote_addr`, etc.) were correctly left untouched, and
confirmed nginx served the built SPA on the custom port.

Documented contract: `BACKEND_URL` must be set **without** a trailing slash. The
`rewrite ^/api/(.*)$ /$1 break;` + bare `proxy_pass ${BACKEND_URL};` pairing strips
the `/api/` prefix correctly either way, but mixing `rewrite ... break` with a
trailing-slash `proxy_pass` target is a well-known nginx footgun (can produce a
doubled slash) — the no-trailing-slash contract sidesteps it entirely.

**3. CORS was already env-var-driven** (fixed in an earlier phase), but is worth
restating here: with the nginx-proxy design, the browser's `/api/*` requests are
same-origin from its perspective (talking to the frontend's own nginx, not
directly to the backend), so CORS doesn't gate the primary traffic path at all.
It still matters for direct access to the backend's own URL (its `/docs` Swagger
UI, or any future separate client), so `CORS_ALLOWED_ORIGINS` should still be
updated with the frontend's real URL once known.

## Known gap: not runtime-verified locally

The full stack was **not** confirmed running end-to-end via `docker-compose up`
locally — the backend container OOM-killed (`OOMKilled: true`, exit 137) on this
host's Docker Desktop VM (~3.83GB allocated, host has only 8GB total RAM). Root
cause almost certainly: a 270MB pickled NetworkX MultiDiGraph typically inflates to
1.5-2.5GB as live Python objects (per-node/edge attribute dict overhead), plus
full-graph GraphSAGE inference over 515K nodes, plus torch/pandas/sklearn's own
baseline footprint. Local (non-Docker) runs throughout this project worked because
macOS lets a bare process use virtual memory/swap dynamically; Docker's Linux VM
enforces a hard memory ceiling with immediate kill instead.

Per user direction, local docker-compose verification was skipped in favor of
proceeding to Render-readiness given the host's RAM constraint. **Both images do
build successfully**, the `$PORT`/`BACKEND_URL` mechanisms were verified in
isolation (above), but the backend's actual peak memory requirement is unverified
— pick a Render instance size with real headroom (see deployment steps) and watch
the deploy logs for OOM on first boot.

## Files added/changed this phase

- `Dockerfile.backend`: shell-form CMD, `$PORT` support
- `Dockerfile.frontend`: copies `nginx.conf.template` into
  `/etc/nginx/templates/`, sets `BACKEND_URL`/`PORT` defaults
- New: `src/frontend/nginx.conf.template` (replaces the old static `nginx.conf`)
- No changes needed to `docker-compose.yml`, `requirements-api.txt`,
  `package.json` — all confirmed already correct/clean for this deploy model
