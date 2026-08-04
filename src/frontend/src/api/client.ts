// All calls hit the real FastAPI backend — no mock data. The backend must
// be running (see project README) for anything here to work.
//
// API_BASE is configurable via VITE_API_BASE_URL (a Vite build-time env
// var, baked into the JS bundle at `npm run build`). Local dev
// (`npm run dev`) defaults to the backend's usual localhost port, matching
// every earlier phase's testing.

import type {
  ExplanationResponse,
  FlaggedAccountsResponse,
  ModelInfoResponse,
  RiskScoreResponse,
  SubgraphResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function getModelInfo() {
  return request<ModelInfoResponse>("/model/info");
}

export function getFlaggedAccounts(page: number, pageSize: number) {
  return request<FlaggedAccountsResponse>(`/accounts/flagged?page=${page}&page_size=${pageSize}`);
}

export function getRiskScore(accountId: string) {
  return request<RiskScoreResponse>(`/accounts/${encodeURIComponent(accountId)}/risk-score`);
}

export function getSubgraph(accountId: string) {
  return request<SubgraphResponse>(`/accounts/${encodeURIComponent(accountId)}/subgraph`);
}

export function getExplanation(accountId: string) {
  return request<ExplanationResponse>(`/accounts/${encodeURIComponent(accountId)}/explanation`);
}

export { ApiError };
