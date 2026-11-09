// Calls the FastAPI backend (must be running, see README).
// Base URL comes from VITE_API_BASE_URL, defaults to localhost:8000.

import type {
  ExplanationResponse,
  FlaggedAccountsResponse,
  FlaggedSortBy,
  ModelInfoResponse,
  RiskScoreResponse,
  SortOrder,
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

export function getFlaggedAccounts(
  page: number,
  pageSize: number,
  sortBy: FlaggedSortBy,
  sortOrder: SortOrder,
  search = "",
) {
  const q = search.trim() ? `&search=${encodeURIComponent(search.trim())}` : "";
  return request<FlaggedAccountsResponse>(
    `/accounts/flagged?page=${page}&page_size=${pageSize}&sort_by=${sortBy}&sort_order=${sortOrder}${q}`,
  );
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
