import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, getExplanation, getRiskScore, getSubgraph } from "../api/client";
import type { ExplanationResponse, RiskScoreResponse, SubgraphResponse } from "../api/types";
import { StatusControl } from "../components/StatusControl";
import { SubgraphView } from "../components/SubgraphView";
import "./AccountDetail.css";

function summarizeCycleLengths(lengths: number[]): string {
  const counts = new Map<number, number>();
  for (const len of lengths) counts.set(len, (counts.get(len) ?? 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([len, count]) => `${len}-cycle ×${count}`)
    .join(", ");
}

export function AccountDetail() {
  const { accountId = "" } = useParams();

  const [risk, setRisk] = useState<RiskScoreResponse | null>(null);
  const [subgraph, setSubgraph] = useState<SubgraphResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [explaining, setExplaining] = useState(true);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setExplaining(true);
    setError(null);
    setRisk(null);
    setSubgraph(null);
    setExplanation(null);
    setExplainError(null);

    Promise.all([getRiskScore(accountId), getSubgraph(accountId)])
      .then(([riskResp, subgraphResp]) => {
        if (cancelled) return;
        setRisk(riskResp);
        setSubgraph(subgraphResp);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : String(e));
        setLoading(false);
        setExplaining(false);
      });

    // Explanation is slow (0.3-3s), so load it separately.
    getExplanation(accountId)
      .then((resp) => {
        if (!cancelled) setExplanation(resp);
      })
      .catch((e: unknown) => {
        // Optional: the rest of the page still works without it.
        if (!cancelled) setExplainError(e instanceof ApiError ? e.message : null);
      })
      .finally(() => {
        if (!cancelled) setExplaining(false);
      });

    return () => {
      cancelled = true;
    };
  }, [accountId]);

  if (loading) return <div className="banner">Loading account {accountId}…</div>;
  if (error) return <div className="banner banner-error">Failed to load: {error}</div>;
  if (!risk || !subgraph) return null;

  const f = risk.key_features;

  return (
    <div className="account-detail">
      <Link to="/" className="back-link">
        ← Back to flagged accounts
      </Link>

      <div className="detail-header">
        <div>
          <h1 className="mono">{risk.account_id}</h1>
          <div className="detail-badges">
            <span className={`label-badge label-${risk.predicted_label}`}>
              predicted: {risk.predicted_label}
            </span>
            {risk.true_label && (
              <span className={`label-badge label-outline label-${risk.true_label}`}>
                true label: {risk.true_label} (demo dataset only)
              </span>
            )}
          </div>
        </div>
        <StatusControl accountId={risk.account_id} />
      </div>

      <div className="detail-grid">
        <div className="panel">
          <h2>Risk Score</h2>
          <div className="score-display">
            <span className="score-value mono">{risk.probability.toFixed(4)}</span>
            <span className="score-threshold">threshold {risk.threshold.toFixed(4)}</span>
          </div>
          <div className="risk-bar-track large">
            <div className="risk-bar-fill" style={{ width: `${risk.probability * 100}%` }} />
            <div className="threshold-marker" style={{ left: `${risk.threshold * 100}%` }} />
          </div>
        </div>

        <div className="panel">
          <h2>Key Features</h2>
          <dl className="feature-list">
            <div><dt>in_degree</dt><dd className="mono">{f.in_degree.toFixed(0)}</dd></div>
            <div><dt>out_degree</dt><dd className="mono">{f.out_degree.toFixed(0)}</dd></div>
            <div><dt>total_degree</dt><dd className="mono">{f.total_degree.toFixed(0)}</dd></div>
            <div><dt>betweenness</dt><dd className="mono">{f.betweenness.toExponential(3)}</dd></div>
            <div><dt>community_illicit_density</dt><dd className="mono">{(f.community_illicit_density * 100).toFixed(2)}%</dd></div>
            <div><dt>community_size</dt><dd className="mono">{f.community_size.toFixed(0)}</dd></div>
            <div><dt>cycle_count</dt><dd className="mono">{f.cycle_count.toFixed(0)}</dd></div>
            <div><dt>transaction_count</dt><dd className="mono">{f.transaction_count.toFixed(0)}</dd></div>
          </dl>
        </div>

        <div className="panel">
          <h2>GNNExplainer</h2>
          {explaining && <p className="muted">Running GNNExplainer for this account…</p>}
          {!explaining && explanation && (
            <>
              <p className="muted">
                In {explanation.cycle_membership_count} detected cycle(s)
                {explanation.cycle_lengths.length > 0 && ` (${summarizeCycleLengths(explanation.cycle_lengths)})`}
                {" · "}explained over a {explanation.explain_hops}-hop, {explanation.explain_subgraph_size}-node subgraph
              </p>
              <h3>Top contributing features</h3>
              <ul className="attribution-list">
                {explanation.top_features.map((tf) => (
                  <li key={tf.feature}>
                    <span className="mono">{tf.feature}</span>
                    <span className="mono attribution-value">{tf.importance.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
              <h3>Top contributing neighbors</h3>
              <ul className="attribution-list">
                {explanation.top_edges.map((te) => (
                  <li key={`${te.source}-${te.target}`}>
                    <span className="mono">{te.source} → {te.target}</span>
                    <span className="mono attribution-value">{te.importance.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {!explaining && !explanation && (
            <p className="muted">{explainError ?? "Explanation unavailable for this account."}</p>
          )}
        </div>
      </div>

      <div className="panel">
        <h2>Local Sub-graph ({subgraph.hops}-hop, {subgraph.node_count} nodes, {subgraph.edge_count} edges)</h2>
        {subgraph.truncated && (
          <p className="muted">Large account: showing only its busiest neighbours.</p>
        )}
        <SubgraphView subgraph={subgraph} explanation={explanation} />
      </div>
    </div>
  );
}
