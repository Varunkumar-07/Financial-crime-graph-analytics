// Interactive counterpart to the Phase 10 explainability PNGs: same color
// language (red = illicit, blue = licit, gold ring = same community, orange
// = high GNNExplainer importance), but zoomable/pannable/clickable.

import cytoscape, { type ElementDefinition, type StylesheetStyle } from "cytoscape";
import coseBilkent from "cytoscape-cose-bilkent";
import { useEffect, useMemo, useRef } from "react";
import { useNavigate } from "react-router-dom";
import type { ExplanationResponse, SubgraphResponse } from "../api/types";
import "./SubgraphView.css";

cytoscape.use(coseBilkent);

interface Props {
  subgraph: SubgraphResponse;
  explanation: ExplanationResponse | null;
}

const IMPORTANCE_HIGHLIGHT_THRESHOLD = 0.3;

export function SubgraphView({ subgraph, explanation }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const navigate = useNavigate();

  const importanceByEdge = useMemo(() => {
    const map = new Map<string, number>();
    for (const e of explanation?.top_edges ?? []) {
      map.set(`${e.source}->${e.target}`, e.importance);
    }
    return map;
  }, [explanation]);

  const elements: ElementDefinition[] = useMemo(() => {
    const nodes: ElementDefinition[] = subgraph.nodes.map((n) => ({
      data: {
        id: n.account_id,
        label: n.account_id,
        illicit: n.is_illicit_labeled,
        center: n.is_center,
        sameCommunity: n.same_community_as_center,
        cycleFlag: n.cycle_flag,
        degree: n.total_degree,
      },
    }));

    const edges: ElementDefinition[] = subgraph.edges.map((e, i) => {
      const importance = importanceByEdge.get(`${e.source}->${e.target}`) ?? 0;
      return {
        data: {
          id: `e${i}`,
          source: e.source,
          target: e.target,
          illicit: e.is_illicit,
          importance,
          txCount: e.transaction_count,
        },
      };
    });

    return [...nodes, ...edges];
  }, [subgraph, importanceByEdge]);

  const stylesheet: StylesheetStyle[] = [
    {
      selector: "node",
      style: {
        "background-color": "#3b82c4",
        label: "data(label)",
        color: "#cbd5e1",
        "font-size": 8,
        "text-valign": "bottom",
        "text-margin-y": 4,
        width: 24,
        height: 24,
        "border-width": 1,
        "border-color": "#334155",
      },
    },
    { selector: "node[?illicit]", style: { "background-color": "#e0405a" } },
    {
      selector: "node[?center]",
      style: { width: 42, height: 42, "border-width": 3, "border-color": "#f5f5f5" },
    },
    {
      selector: "node[?sameCommunity][!center]",
      style: { "border-width": 2.5, "border-color": "#e0b13d" },
    },
    {
      selector: "edge",
      style: {
        width: 1.2,
        "line-color": "#3a4150",
        "target-arrow-color": "#3a4150",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        opacity: 0.85,
      },
    },
    {
      selector: "edge[?illicit]",
      style: { "line-color": "#e0405a", "target-arrow-color": "#e0405a", width: 2.2 },
    },
    {
      selector: `edge[importance >= ${IMPORTANCE_HIGHLIGHT_THRESHOLD}][!illicit]`,
      style: {
        "line-color": "#e08a3d",
        "target-arrow-color": "#e08a3d",
        width: "mapData(importance, 0, 1, 2, 6)",
      },
    },
  ];

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: stylesheet,
      layout: { name: "cose-bilkent", animate: false, nodeRepulsion: 6000 } as unknown as cytoscape.LayoutOptions,
      minZoom: 0.2,
      maxZoom: 4,
    });
    cyRef.current = cy;

    cy.on("tap", "node", (evt) => {
      const id = evt.target.id();
      if (id !== subgraph.account_id) navigate(`/accounts/${encodeURIComponent(id)}`);
    });

    return () => cy.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements]);

  return (
    <div className="subgraph-view">
      <div className="subgraph-legend">
        <span><i className="dot illicit" /> Illicit-labeled</span>
        <span><i className="dot licit" /> Licit-labeled</span>
        <span><i className="ring center" /> Target account</span>
        <span><i className="ring community" /> Same community</span>
        <span><i className="line importance" /> High GNNExplainer importance</span>
      </div>
      <div ref={containerRef} className="cy-container" />
      <p className="subgraph-hint">
        Scroll to zoom, drag to pan, click a neighbor node to jump to its own account view.
      </p>
    </div>
  );
}
