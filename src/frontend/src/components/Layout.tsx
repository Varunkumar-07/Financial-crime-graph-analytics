import { useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";
import { getModelInfo } from "../api/client";
import type { ModelInfoResponse } from "../api/types";
import "./Layout.css";

// Metrics come live from /model/info so they stay correct after a retrain.
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
