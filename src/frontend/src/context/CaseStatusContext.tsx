// Case statuses, saved in this browser's localStorage (no backend storage).

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { CaseStatus } from "../api/types";

const STORAGE_KEY = "fcg-case-statuses";

function loadStatuses(): Record<string, CaseStatus> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, CaseStatus>) : {};
  } catch {
    return {};
  }
}

interface CaseStatusContextValue {
  getStatus: (accountId: string) => CaseStatus;
  setStatus: (accountId: string, status: CaseStatus) => void;
  // Complete for reviewing/cleared/escalated (only set by a click).
  // Not for "flagged", which is everyone's default.
  getAccountsByStatus: (status: CaseStatus) => string[];
}

const CaseStatusContext = createContext<CaseStatusContextValue | null>(null);

export function CaseStatusProvider({ children }: { children: ReactNode }) {
  const [statuses, setStatuses] = useState<Record<string, CaseStatus>>(loadStatuses);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(statuses));
    } catch {
      // storage unavailable (e.g. private mode): statuses just won't persist
    }
  }, [statuses]);

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
