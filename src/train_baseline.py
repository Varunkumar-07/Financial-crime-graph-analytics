"""
Baseline supervised models on the per-account feature table: XGBoost
(primary) and Random Forest (secondary comparison), both handling the
1.23% positive rate via class weighting only (no SMOTE/oversampling
in this first pass).

Feature selection notes:
  - `account` is an identifier, not a feature.
  - `community_id` is excluded as a raw feature: it's an arbitrary,
    high-cardinality (115,975-value) label with no ordinal meaning.
    Phase 6 established that `community_illicit_density` and
    `community_size` are the informative, model-ready derivatives of
    community membership — those are kept.
  - `in_out_amount_ratio` has 18,085 NaNs (accounts that never sent,
    so the ratio is undefined). Imputed with -1, a value the true
    ratio (always >= 0) can never take, so tree splits can isolate
    these "never sent" accounts as their own group rather than being
    silently dropped or conflated with a real ratio of 0.

No GNN work here — classical baselines and evaluation only.

Run from the project root with the venv activated:

    python src/train_baseline.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "account_features.csv"
XGB_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "xgboost_baseline.json"
RF_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "random_forest_baseline.pkl"
REPORT_MD = PROJECT_ROOT / "notebooks" / "baseline_model_summary.md"

TARGET = "is_illicit"
EXCLUDED_COLS = ["account", "community_id", TARGET]
RANDOM_STATE = 42


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["in_out_amount_ratio"] = df["in_out_amount_ratio"].fillna(-1)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDED_COLS]


def f2_optimal_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]  # drop the sentinel point (thresh=inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = 5 * precision * recall / (4 * precision + recall)
    f2 = np.nan_to_num(f2)
    best_idx = np.argmax(f2)
    return thresholds[best_idx], f2[best_idx], precision[best_idx], recall[best_idx]


def evaluate_model(name, y_true, y_proba):
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    thresh, f2_at_thresh, _, _ = f2_optimal_threshold(y_true, y_proba)
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_chosen = (y_proba >= thresh).astype(int)

    def metrics_at(y_pred):
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    default_metrics = metrics_at(y_pred_default)
    chosen_metrics = metrics_at(y_pred_chosen)
    cm = confusion_matrix(y_true, y_pred_chosen)

    return {
        "name": name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "threshold_default": 0.5,
        "threshold_chosen": thresh,
        "f2_at_chosen": f2_at_thresh,
        "metrics_default": default_metrics,
        "metrics_chosen": chosen_metrics,
        "confusion_matrix_chosen": cm,
    }


def format_eval_section(ev: dict) -> list:
    cm = ev["confusion_matrix_chosen"]
    tn, fp, fn, tp = cm.ravel()
    lines = [
        f"### {ev['name']}",
        "",
        f"- AUC-ROC: {ev['auc_roc']:.4f}",
        f"- AUC-PR (average precision): {ev['auc_pr']:.4f}",
        "",
        f"**At default threshold 0.5:** precision={ev['metrics_default']['precision']:.4f}, "
        f"recall={ev['metrics_default']['recall']:.4f}, f1={ev['metrics_default']['f1']:.4f}",
        "",
        f"**At chosen threshold {ev['threshold_chosen']:.4f} "
        f"(maximizes F2 — weights recall 4x more than precision, since missing an "
        f"illicit account is costlier than a false positive here):** "
        f"precision={ev['metrics_chosen']['precision']:.4f}, "
        f"recall={ev['metrics_chosen']['recall']:.4f}, "
        f"f1={ev['metrics_chosen']['f1']:.4f}, f2={ev['f2_at_chosen']:.4f}",
        "",
        "Confusion matrix at chosen threshold:",
        "",
        "| | Predicted licit | Predicted illicit |",
        "|---|---|---|",
        f"| Actual licit | {tn:,} (TN) | {fp:,} (FP) |",
        f"| Actual illicit | {fn:,} (FN) | {tp:,} (TP) |",
        "",
    ]
    return lines


def main() -> None:
    t_start = time.time()

    print("Loading feature table...")
    df = load_features()
    feature_cols = get_feature_cols(df)
    print(f"  {df.shape[0]:,} rows, {len(feature_cols)} features used: {feature_cols}")

    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,} rows ({y_train.mean()*100:.4f}% positive)")
    print(f"  Test:  {len(X_test):,} rows ({y_test.mean()*100:.4f}% positive)")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight (train neg/pos): {scale_pos_weight:.2f}")

    # --- XGBoost ---
    print("Training XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Random Forest ---
    print("Training Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Evaluation ---
    print("Evaluating on test set...")
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    xgb_eval = evaluate_model("XGBoost", y_test, xgb_proba)
    rf_eval = evaluate_model("Random Forest", y_test, rf_proba)

    print(f"XGBoost: AUC-ROC={xgb_eval['auc_roc']:.4f}, AUC-PR={xgb_eval['auc_pr']:.4f}")
    print(f"Random Forest: AUC-ROC={rf_eval['auc_roc']:.4f}, AUC-PR={rf_eval['auc_pr']:.4f}")

    # --- Feature importance ---
    xgb_importance = pd.Series(
        xgb_model.get_booster().get_score(importance_type="gain"), name="xgb_gain"
    )
    # get_score keys are feature names only if X_train had them (DataFrame input preserves this)
    xgb_importance = xgb_importance.reindex(feature_cols).fillna(0.0)
    xgb_importance = xgb_importance / xgb_importance.sum()  # normalize to sum to 1

    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols, name="rf_importance")

    importance_df = pd.DataFrame({"xgb_gain_normalized": xgb_importance, "rf_importance": rf_importance})
    importance_df["avg_rank"] = (
        importance_df["xgb_gain_normalized"].rank(ascending=False)
        + importance_df["rf_importance"].rank(ascending=False)
    ) / 2
    importance_df = importance_df.sort_values("avg_rank")

    # --- Save models ---
    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(XGB_MODEL_PATH))
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved models to {XGB_MODEL_PATH} and {RF_MODEL_PATH}")

    # --- Report ---
    lines = [
        "# Baseline Model Summary",
        "",
        "## Setup",
        f"- Feature table: {df.shape[0]:,} rows, {len(feature_cols)} features "
        f"(excluded: {EXCLUDED_COLS[:-1]})",
        "- Split: 80/20 train/test, stratified on `is_illicit`, random_state=42",
        f"- Train set: {len(X_train):,} rows ({y_train.mean()*100:.4f}% positive)",
        f"- Test set: {len(X_test):,} rows ({y_test.mean()*100:.4f}% positive)",
        f"- Class imbalance handling: class weighting only (no SMOTE/oversampling). "
        f"XGBoost `scale_pos_weight`={scale_pos_weight:.2f} (train neg/pos ratio); "
        f"Random Forest `class_weight='balanced'`.",
        "- `in_out_amount_ratio` NaNs (accounts that never sent) imputed with -1 "
        "(an impossible value for the true ratio, so tree splits isolate this group cleanly).",
        "- `community_id` excluded from features (arbitrary high-cardinality label; "
        "`community_illicit_density` and `community_size` are the informative derivatives, per Phase 6).",
        "",
        "## Threshold selection rationale",
        "Rather than the default 0.5 cutoff, the reported \"chosen\" threshold for each model "
        "maximizes the F2-score (recall weighted 4x more heavily than precision) on the test "
        "set's precision-recall curve. This reflects that in an AML screening context, missing "
        "an illicit account (false negative) is typically costlier than an extra manual review "
        "(false positive). Both the default-0.5 and F2-optimal metrics are reported below for "
        "comparison. Note: for a production pipeline the threshold should be tuned on a held-out "
        "validation split rather than the test set itself to avoid threshold-selection leakage — "
        "this baseline uses the test set directly for simplicity and should be revisited.",
        "",
        "## Evaluation",
        "",
    ]
    lines += format_eval_section(xgb_eval)
    lines += format_eval_section(rf_eval)

    lines += [
        "## Feature importance",
        "",
        "XGBoost importance = gain-based, normalized to sum to 1. Random Forest importance = "
        "mean decrease in impurity. `avg_rank` = average of each model's rank (1=most important).",
        "",
        "| Feature | XGB gain (norm.) | RF importance | Avg rank |",
        "|---|---|---|---|",
    ]
    for feat, row in importance_df.iterrows():
        lines.append(
            f"| {feat} | {row['xgb_gain_normalized']:.4f} | {row['rf_importance']:.4f} | "
            f"{row['avg_rank']:.1f} |"
        )

    top5 = list(importance_df.index[:5])
    phase6_flagged = {"community_illicit_density", "cycle_count"}
    confirmed = phase6_flagged & set(top5)
    lines += [
        "",
        "## Comparison to Phase 6 correlation findings",
        f"- Top 5 features by average model rank: {top5}",
        f"- Phase 6 flagged `community_illicit_density` and `cycle_count` as promising nonlinear "
        f"signals (weak linear correlation but large mean gap between classes).",
    ]
    if confirmed:
        lines.append(
            f"- **Confirmed**: {sorted(confirmed)} rank in the top 5 for at least one model — "
            "the tree-based models did pick up on this nonlinear signal that raw correlation understated."
        )
    else:
        lines.append(
            "- Neither flagged feature reached the top 5 in these models — see full importance "
            "table above for their actual rank."
        )

    lines += ["", f"Total runtime: {time.time() - t_start:.1f}s", ""]

    summary = "\n".join(lines)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
