// Mirrors the Pydantic schemas in src/api/schemas/*.py exactly — keep in sync.

export interface KeyFeatures {
  in_degree: number;
  out_degree: number;
  total_degree: number;
  betweenness: number;
  community_id: number;
  community_size: number;
  community_illicit_density: number;
  cycle_count: number;
  cycle_flag: number;
  total_sent: number;
  total_received: number;
  transaction_count: number;
  avg_transaction_amount: number;
  in_out_amount_ratio: number;
}

export interface RiskScoreResponse {
  account_id: string;
  probability: number;
  predicted_label: "illicit" | "licit";
  threshold: number;
  true_label: "illicit" | "licit" | null;
  key_features: KeyFeatures;
}

export interface FlaggedAccount {
  account_id: string;
  probability: number;
  predicted_label: "illicit" | "licit";
}

export interface FlaggedAccountsResponse {
  page: number;
  page_size: number;
  total_flagged: number;
  total_pages: number;
  threshold: number;
  accounts: FlaggedAccount[];
}

export interface SubgraphNode {
  account_id: string;
  is_center: boolean;
  is_illicit_labeled: boolean;
  community_id: number;
  same_community_as_center: boolean;
  total_degree: number;
  cycle_flag: boolean;
}

export interface SubgraphEdge {
  source: string;
  target: string;
  is_illicit: boolean;
  transaction_count: number;
}

export interface SubgraphResponse {
  account_id: string;
  hops: number;
  node_count: number;
  edge_count: number;
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
}

export interface FeatureAttribution {
  feature: string;
  importance: number;
}

export interface EdgeAttribution {
  source: string;
  target: string;
  importance: number;
}

export interface ExplanationResponse {
  account_id: string;
  probability: number;
  predicted_label: "illicit" | "licit";
  top_features: FeatureAttribution[];
  top_edges: EdgeAttribution[];
  cycle_membership_count: number;
  cycle_lengths: number[];
  explain_hops: number;
  explain_subgraph_size: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  n_accounts: number;
  n_cycles_cached: number;
  threshold: number;
}

export interface ModelInfoResponse {
  model_name: string;
  auc_roc: number;
  auc_pr: number;
  threshold: number;
  n_test_accounts: number;
}

export type CaseStatus = "flagged" | "reviewing" | "cleared" | "escalated";
