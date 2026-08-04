import type { CaseStatus } from "../api/types";
import { useCaseStatus } from "../context/CaseStatusContext";
import "./StatusControl.css";

const OPTIONS: CaseStatus[] = ["flagged", "reviewing", "cleared", "escalated"];

export function StatusControl({ accountId }: { accountId: string }) {
  const { getStatus, setStatus } = useCaseStatus();
  const current = getStatus(accountId);

  return (
    <select
      className={`status-select status-${current}`}
      value={current}
      onChange={(e) => setStatus(accountId, e.target.value as CaseStatus)}
      onClick={(e) => e.stopPropagation()}
    >
      {OPTIONS.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}
