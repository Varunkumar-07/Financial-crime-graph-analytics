"""
Exploratory data analysis for the IBM AML synthetic transaction dataset
(HI-Small_Trans.csv).

Read-only EDA: no cleaning, filtering, or graph construction here.
Run from the project root with the venv activated:

    python src/eda.py

Writes a Markdown summary to notebooks/eda_summary.md.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"
SUMMARY_MD = PROJECT_ROOT / "notebooks" / "eda_summary.md"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # The source CSV has two columns both named "Account" (sender, receiver);
    # pandas auto-suffixes the second as "Account.1". Rename for clarity.
    df = df.rename(columns={"Account": "From Account", "Account.1": "To Account"})
    return df


def build_summary(df: pd.DataFrame) -> str:
    total_rows = len(df)
    n_sender_accounts = df["From Account"].nunique()
    n_receiver_accounts = df["To Account"].nunique()

    schema_lines = [f"- `{col}`: {dtype}" for col, dtype in df.dtypes.items()]

    laundering_col = "Is Laundering"
    class_counts = df[laundering_col].value_counts().sort_index()
    class_pct = df[laundering_col].value_counts(normalize=True).sort_index() * 100
    class_lines = []
    for label in class_counts.index:
        name = "Illicit (1)" if label == 1 else "Licit (0)"
        class_lines.append(
            f"- {name}: {class_counts[label]:,} ({class_pct[label]:.4f}%)"
        )

    timestamps = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    time_min, time_max = timestamps.min(), timestamps.max()

    null_counts = df.isnull().sum()
    null_lines = [f"- `{col}`: {count:,}" for col, count in null_counts.items()]

    lines = [
        "# EDA Summary — HI-Small_Trans.csv",
        "",
        "## Row count",
        f"- Total transactions: {total_rows:,}",
        "",
        "## Unique accounts",
        f"- Unique sender accounts (`From Account`): {n_sender_accounts:,}",
        f"- Unique receiver accounts (`To Account`): {n_receiver_accounts:,}",
        "",
        "## Column schema",
        *schema_lines,
        "",
        "## Class balance (`Is Laundering`)",
        *class_lines,
        "",
        "## Time range",
        f"- Start: {time_min}",
        f"- End: {time_max}",
        "",
        "## Missing values per column",
        *null_lines,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    summary = build_summary(df)
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
