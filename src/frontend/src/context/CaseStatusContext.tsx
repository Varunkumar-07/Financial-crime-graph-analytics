// Client-side only, in-memory case status tracking. Resets on page reload —
// no backend persistence yet (explicitly out of scope for this phase).

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { CaseStatus } from "../api/types";

interface CaseStatusContextValue {
  getStatus: (accountId: string) => CaseStatus;
  setStatus: (accountId: string, status: CaseStatus) => void;
  // Only meaningful for "reviewing" | "cleared" | "escalated" — those are
  // never a default, so every account with one of them is guaranteed to be
  // in `statuses` already. This is a COMPLETE list across the whole flagged
  // dataset, not just whatever's currently loaded on screen — unlike
  // "flagged" (the default for everyone, including accounts never seen this
  // session, which this map has no way to know about). See Dashboard.tsx's
  // status-filter split for why that distinction matters.
  getAccountsByStatus: (status: CaseStatus) => string[];
}

const CaseStatusContext = createContext<CaseStatusContextValue | null>(null);

export function CaseStatusProvider({ children }: { children: ReactNode }) {
  const [statuses, setStatuses] = useState<Record<string, CaseStatus>>({});

  const value = useMemo<CaseStatusContextValue>(
    () => ({
      getStatus: (accountId) => statuses[accountId] ?? "flagged",
      setStatus: (accountId, status) =>
        setStatuses((prev) => ({ ...prev, [accountId]: status })),
      getAccountsByStatus: (status) =>
        Object.entries(statuses)
          .filter(([, s]) => s === status)
          .map(([accountId]) => accountId),
    }),
    [statuses],
  );

  return <CaseStatusContext.Provider value={value}>{children}</CaseStatusContext.Provider>;
}

export function useCaseStatus() {
  const ctx = useContext(CaseStatusContext);
  if (!ctx) throw new Error("useCaseStatus must be used within CaseStatusProvider");
  return ctx;
}
