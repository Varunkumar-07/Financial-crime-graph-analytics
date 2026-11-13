import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, getFlaggedAccounts, getRiskScore } from "../api/client";
import type {
  CaseStatus,
  FlaggedAccountsResponse,
  FlaggedSortBy,
  RiskScoreResponse,
  SortOrder,
} from "../api/types";
import { StatusControl } from "../components/StatusControl";
import { useCaseStatus } from "../context/CaseStatusContext";
import "./Dashboard.css";

const PAGE_SIZE = 25;
const PAGE_WINDOW_RADIUS = 2; // page buttons on each side

const SORT_LABELS: Record<FlaggedSortBy, string> = {
  risk_score: "Risk Score",
  degree: "Degree",
  community_illicit_pct: "Community Illicit %",
  cycle_count: "Cycles",
};

// Statuses only set by a click, so we know every account that has them.
const FULLY_TRACKED_STATUSES: CaseStatus[] = ["reviewing", "cleared", "escalated"];

interface DisplayRow {
  account_id: string;
  probability: number;
  total_degree: number;
  community_illicit_density: number;
  cycle_count: number;
}

const SORT_VALUE: Record<FlaggedSortBy, (r: DisplayRow) => number> = {
  risk_score: (r) => r.probability,
  degree: (r) => r.total_degree,
  community_illicit_pct: (r) => r.community_illicit_density,
  cycle_count: (r) => r.cycle_count,
};

function pageWindow(current: number, total: number): number[] {
  const desired = PAGE_WINDOW_RADIUS * 2 + 1;
  let start = Math.max(1, current - PAGE_WINDOW_RADIUS);
  let end = Math.min(total, current + PAGE_WINDOW_RADIUS);
  if (end - start + 1 < desired) {
    if (start === 1) end = Math.min(total, start + desired - 1);
    else if (end === total) start = Math.max(1, end - desired + 1);
  }
  const pages: number[] = [];
  for (let p = start; p <= end; p++) pages.push(p);
  return pages;
}

export function Dashboard() {
  const navigate = useNavigate();
  const { getStatus, getAccountsByStatus } = useCaseStatus();

  const [page, setPage] = useState(1);
  // Sorting happens on the backend over all flagged accounts, so each
  // sort/page change re-fetches.
  const [sortBy, setSortBy] = useState<FlaggedSortBy>("risk_score");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [listing, setListing] = useState<FlaggedAccountsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [pageInput, setPageInput] = useState("1");

  // Reviewing/Cleared/Escalated use the full list; others use server paging.
  const isTouchedMode = FULLY_TRACKED_STATUSES.includes(statusFilter as CaseStatus);
  const touchedIds = useMemo(
    () => (isTouchedMode ? getAccountsByStatus(statusFilter as CaseStatus) : []),
    [isTouchedMode, statusFilter, getAccountsByStatus],
  );
  const [touchedRows, setTouchedRows] = useState<DisplayRow[]>([]);
  const [touchedLoading, setTouchedLoading] = useState(false);
  const [touchedError, setTouchedError] = useState<string | null>(null);
  const [touchedWarning, setTouchedWarning] = useState<string | null>(null);

  // Wait for typing to pause before searching on the backend.
  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    // Status views list marked accounts directly; no need for the flagged page.
    if (isTouchedMode) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    getFlaggedAccounts(page, PAGE_SIZE, sortBy, sortOrder, debouncedSearch)
      .then((resp) => {
        if (cancelled) return;
        setListing(resp);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [page, sortBy, sortOrder, debouncedSearch, isTouchedMode]);

  useEffect(() => {
    setPageInput(String(page));
  }, [page]);

  useEffect(() => {
    if (!isTouchedMode) return;
    let cancelled = false;
    setTouchedLoading(true);
    setTouchedError(null);
    setTouchedWarning(null);

    // allSettled: one missing account shouldn't hide all the others.
    Promise.allSettled(touchedIds.map((id) => getRiskScore(id)))
      .then((settled) => {
        if (cancelled) return;
        const results = settled
          .filter((s): s is PromiseFulfilledResult<RiskScoreResponse> => s.status === "fulfilled")
          .map((s) => s.value);
        const failed = settled.length - results.length;
        if (failed > 0) setTouchedWarning(`${failed} marked account(s) could not be loaded.`);
        setTouchedRows(
          results.map((r) => ({
            account_id: r.account_id,
            probability: r.probability,
            total_degree: r.key_features.total_degree,
            community_illicit_density: r.key_features.community_illicit_density,
            cycle_count: r.key_features.cycle_count,
          })),
        );
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setTouchedError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : String(e));
      })
      .finally(() => {
        if (!cancelled) setTouchedLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // join(",") keeps the dependency stable across renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTouchedMode, touchedIds.join(",")]);

  const rows = useMemo((): DisplayRow[] => {
    if (isTouchedMode) {
      let filtered = touchedRows;
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        filtered = filtered.filter((a) => a.account_id.toLowerCase().includes(q));
      }
      // Full set, so client-side sorting is fine here.
      const getVal = SORT_VALUE[sortBy];
      return [...filtered].sort((a, b) => {
        const diff = getVal(a) - getVal(b);
        return sortOrder === "asc" ? diff : -diff;
      });
    }

    if (!listing) return [];
    // Search is already applied on the backend.
    let filtered: DisplayRow[] = listing.accounts;
    if (statusFilter === "flagged") {
      // Current page only: we can't know "flagged" status for unloaded accounts.
      filtered = filtered.filter((a) => getStatus(a.account_id) === "flagged");
    }
    return filtered;
  }, [isTouchedMode, touchedRows, listing, search, statusFilter, sortBy, sortOrder, getStatus]);

  function toggleSort(key: FlaggedSortBy) {
    if (sortBy === key) {
      setSortOrder((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(key);
      setSortOrder("desc");
    }
    setPage(1); // back to first page
  }

  function sortArrow(key: FlaggedSortBy) {
    if (sortBy !== key) return "";
    return sortOrder === "asc" ? " ▲" : " ▼";
  }

  function jumpToPage() {
    const n = Number.parseInt(pageInput, 10);
    const totalPages = listing?.total_pages ?? 1;
    if (!Number.isFinite(n)) {
      setPageInput(String(page));
      return;
    }
    setPage(Math.min(Math.max(1, n), totalPages));
  }

  const totalPages = listing?.total_pages ?? 1;
  const nearbyPages = pageWindow(page, totalPages);
  const showLoading = isTouchedMode ? touchedLoading : loading;
  const showError = isTouchedMode ? touchedError : error;

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div>
          <h1>Flagged Accounts</h1>
          {isTouchedMode ? (
            <p className="dashboard-subtitle">
              Showing all <strong>{rows.length}</strong> accounts you marked{" "}
              <strong>{statusFilter}</strong> (from any page, flagged or not).
            </p>
          ) : (
            listing && (
              <p className="dashboard-subtitle">
                {listing.total_flagged.toLocaleString()} accounts above threshold{" "}
                {debouncedSearch.trim() && <>matching <code>{debouncedSearch.trim()}</code>{" "}</>}
                <code>{listing.threshold.toFixed(4)}</code> — sorted by{" "}
                <strong>{SORT_LABELS[listing.sort_by]}</strong> ({listing.sort_order}) — page{" "}
                {listing.page} of {listing.total_pages.toLocaleString()}
              </p>
            )
          )}
        </div>
        <div className="dashboard-controls">
          <input
            className="search-input"
            placeholder={
              isTouchedMode ? "Filter these results by account ID..." : "Search flagged accounts by ID..."
            }
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All statuses</option>
            <option value="flagged">Flagged</option>
            <option value="reviewing">Reviewing</option>
            <option value="cleared">Cleared</option>
            <option value="escalated">Escalated</option>
          </select>
        </div>
      </div>

      <p className="status-filter-note">
        Reviewing / Cleared / Escalated show every account you have marked with that status.
        Flagged only checks the current page (it's everyone's default, so this can't be a
        complete search).
      </p>

      {showError && <div className="banner banner-error">Failed to load: {showError}</div>}
      {showLoading && <div className="banner">Loading from live backend…</div>}
      {isTouchedMode && !showLoading && touchedWarning && (
        <div className="banner banner-error">{touchedWarning}</div>
      )}

      {!showLoading && !showError && (
        <>
          <table className="accounts-table">
            <thead>
              <tr>
                <th>Account</th>
                <th className="sortable" onClick={() => toggleSort("risk_score")}>
                  Risk Score{sortArrow("risk_score")}
                </th>
                <th className="sortable" onClick={() => toggleSort("degree")}>
                  Degree{sortArrow("degree")}
                </th>
                <th className="sortable" onClick={() => toggleSort("community_illicit_pct")}>
                  Community Illicit %{sortArrow("community_illicit_pct")}
                </th>
                <th className="sortable" onClick={() => toggleSort("cycle_count")}>
                  Cycles{sortArrow("cycle_count")}
                </th>
                <th>Case Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((account) => (
                <tr key={account.account_id} onClick={() => navigate(`/accounts/${encodeURIComponent(account.account_id)}`)}>
                  <td className="mono">{account.account_id}</td>
                  <td>
                    <div className="risk-cell">
                      <div className="risk-bar-track">
                        <div className="risk-bar-fill" style={{ width: `${account.probability * 100}%` }} />
                      </div>
                      <span className="mono">{account.probability.toFixed(4)}</span>
                    </div>
                  </td>
                  <td className="mono">{account.total_degree.toFixed(0)}</td>
                  <td className="mono">{(account.community_illicit_density * 100).toFixed(2)}%</td>
                  <td className="mono">{account.cycle_count.toFixed(0)}</td>
                  <td>
                    <StatusControl accountId={account.account_id} />
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={6} className="empty-row">
                    {isTouchedMode
                      ? `No accounts are currently marked ${statusFilter}.`
                      : "No flagged accounts match the current filters."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          {!isTouchedMode && (
            <div className="pagination">
              <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                ← Previous
              </button>

              {nearbyPages[0] > 1 && (
                <>
                  <button onClick={() => setPage(1)}>1</button>
                  {nearbyPages[0] > 2 && <span className="pagination-ellipsis">…</span>}
                </>
              )}
              {nearbyPages.map((p) => (
                <button key={p} className={p === page ? "pagination-current" : ""} onClick={() => setPage(p)}>
                  {p}
                </button>
              ))}
              {nearbyPages[nearbyPages.length - 1] < totalPages && (
                <>
                  {nearbyPages[nearbyPages.length - 1] < totalPages - 1 && (
                    <span className="pagination-ellipsis">…</span>
                  )}
                  <button onClick={() => setPage(totalPages)}>{totalPages.toLocaleString()}</button>
                </>
              )}

              <button disabled={!listing || page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                Next →
              </button>

              <span className="pagination-jump">
                Go to page{" "}
                <input
                  type="number"
                  min={1}
                  max={totalPages}
                  className="page-jump-input"
                  value={pageInput}
                  onChange={(e) => setPageInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") jumpToPage();
                  }}
                />
                <button onClick={jumpToPage}>Go</button>
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
