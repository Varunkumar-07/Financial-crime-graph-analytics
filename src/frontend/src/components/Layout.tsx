import { useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";
import { getModelInfo } from "../api/client";
import type { ModelInfoResponse } from "../api/types";
import "./Layout.css";

// Model metrics are fetched from GET /model/info, computed live by the
// backend from whatever model is actually loaded — never hardcoded here.
// A retrain changes these numbers, and this header must not silently go
// stale relative to the model actually serving predictions (see
// notebooks/integration_test_summary.md for the incident that caused this).
export function Layout() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setFailed(true));
  }, []);

  let subtitle = "GraphSAGE risk model";
  if (modelInfo) {
    subtitle += ` · AUC-ROC ${modelInfo.auc_roc.toFixed(3)} · AUC-PR ${modelInfo.auc_pr.toFixed(3)}`;
  } else if (!failed) {
    subtitle += " · loading metrics…";
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <Link to="/" className="app-title">
          <span className="app-title-mark">FCG</span>
          Financial Crime Graph Analytics
        </Link>
        <span className="app-subtitle">{subtitle}</span>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
