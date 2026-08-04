import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, getFlaggedAccounts, getRiskScore } from "../api/client";
import type { FlaggedAccountsResponse, RiskScoreResponse } from "../api/types";
import { StatusControl } from "../components/StatusControl";
import { useCaseStatus } from "../context/CaseStatusContext";
import "./Dashboard.css";

const PAGE_SIZE = 25;

type SortKey = "probability" | "total_degree" | "community_illicit_density" | "cycle_count";

export function Dashboard() {
  const navigate = useNavigate();
  const { getStatus } = useCaseStatus();

  const [page, setPage] = useState(1);
  const [listing, setListing] = useState<FlaggedAccountsResponse | null>(null);
  const [details, setDetails] = useState<Record<string, RiskScoreResponse>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("probability");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getFlaggedAccounts(page, PAGE_SIZE)
      .then(async (resp) => {
        if (cancelled) return;
        setListing(resp);
        const entries = await Promise.all(
          resp.accounts.map(async (a) => {
            const detail = await getRiskScore(a.account_id);
            return [a.account_id, detail] as const;
          }),
        );
        if (cancelled) return;
        setDetails(Object.fromEntries(entries));
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
  }, [page]);

  const rows = useMemo(() => {
    if (!listing) return [];
    let filtered = listing.accounts.filter((a) => details[a.account_id]);

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      filtered = filtered.filter((a) => a.account_id.toLowerCase().includes(q));
    }
    if (statusFilter !== "all") {
      filtered = filtered.filter((a) => getStatus(a.account_id) === statusFilter);
    }

    const withDetail = filtered.map((a) => ({ account: a, detail: details[a.account_id] }));
    withDetail.sort((x, y) => {
      const getVal = (d: RiskScoreResponse) =>
        sortKey === "probability" ? d.probability : d.key_features[sortKey];
      const diff = getVal(x.detail) - getVal(y.detail);
      return sortDir === "asc" ? diff : -diff;
    });
    return withDetail;
  }, [listing, details, search, statusFilter, sortKey, sortDir, getStatus]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function sortArrow(key: SortKey) {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " ▲" : " ▼";
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div>
          <h1>Flagged Accounts</h1>
          {listing && (
            <p className="dashboard-subtitle">
              {listing.total_flagged.toLocaleString()} accounts above threshold{" "}
              <code>{listing.threshold.toFixed(4)}</code> — page {listing.page} of{" "}
              {listing.total_pages.toLocaleString()}
            </p>
          )}
        </div>
        <div className="dashboard-controls">
          <input
            className="search-input"
            placeholder="Filter this page by account ID..."
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

      {error && <div className="banner banner-error">Failed to load: {error}</div>}
      {loading && <div className="banner">Loading from live backend…</div>}

      {!loading && !error && (
        <>
          <table className="accounts-table">
            <thead>
              <tr>
                <th>Account</th>
                <th className="sortable" onClick={() => toggleSort("probability")}>
                  Risk Score{sortArrow("probability")}
                </th>
                <th className="sortable" onClick={() => toggleSort("total_degree")}>
                  Degree{sortArrow("total_degree")}
                </th>
                <th className="sortable" onClick={() => toggleSort("community_illicit_density")}>
                  Community Illicit %{sortArrow("community_illicit_density")}
                </th>
                <th className="sortable" onClick={() => toggleSort("cycle_count")}>
                  Cycles{sortArrow("cycle_count")}
                </th>
                <th>Case Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ account, detail }) => (
                <tr key={account.account_id} onClick={() => navigate(`/accounts/${account.account_id}`)}>
                  <td className="mono">{account.account_id}</td>
                  <td>
                    <div className="risk-cell">
                      <div className="risk-bar-track">
                        <div className="risk-bar-fill" style={{ width: `${detail.probability * 100}%` }} />
                      </div>
                      <span className="mono">{detail.probability.toFixed(4)}</span>
                    </div>
                  </td>
                  <td className="mono">{detail.key_features.total_degree.toFixed(0)}</td>
                  <td className="mono">{(detail.key_features.community_illicit_density * 100).toFixed(2)}%</td>
                  <td className="mono">{detail.key_features.cycle_count.toFixed(0)}</td>
                  <td>
                    <StatusControl accountId={account.account_id} />
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={6} className="empty-row">
                    No accounts match the current filters on this page.
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          <div className="pagination">
            <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              ← Previous
            </button>
            <span>
              Page {page} of {listing?.total_pages.toLocaleString()}
            </span>
            <button
              disabled={!listing || page >= listing.total_pages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
