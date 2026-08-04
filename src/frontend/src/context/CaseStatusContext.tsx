// Client-side only, in-memory case status tracking. Resets on page reload —
// no backend persistence yet (explicitly out of scope for this phase).

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { CaseStatus } from "../api/types";

interface CaseStatusContextValue {
  getStatus: (accountId: string) => CaseStatus;
  setStatus: (accountId: string, status: CaseStatus) => void;
}

const CaseStatusContext = createContext<CaseStatusContextValue | null>(null);

export function CaseStatusProvider({ children }: { children: ReactNode }) {
  const [statuses, setStatuses] = useState<Record<string, CaseStatus>>({});

  const value = useMemo<CaseStatusContextValue>(
    () => ({
      getStatus: (accountId) => statuses[accountId] ?? "flagged",
      setStatus: (accountId, status) =>
        setStatuses((prev) => ({ ...prev, [accountId]: status })),
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
