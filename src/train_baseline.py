"""
Trains the XGBoost and Random Forest baselines (class weighting for the
1.23% positive rate).

- `account` and `community_id` are dropped (ID / arbitrary label).
- Missing `in_out_amount_ratio` (never sent) is set to -1.

Usage: python src/train_baseline.py
"""

import json
import pickle
import time

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import EXCLUDED_COLS, PROJECT_ROOT, TARGET, f2_optimal_threshold, get_feature_cols, load_features
from splits import RANDOM_STATE, split_indices

XGB_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "xgboost_baseline.json"
RF_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "random_forest_baseline.pkl"
REPORT_MD = PROJECT_ROOT / "notebooks" / "baseline_model_summary.md"
METRICS_JSON = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"


def make_xgboost(scale_pos_weight):
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model(name, y_true, y_proba, y_val, val_proba):
    """Test metrics, using the F2 threshold picked on validation."""
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    thresh, f2_at_thresh, _, _ = f2_optimal_threshold(y_val, val_proba)
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
        f"(maximizes F2 on the validation set — recall weighted 4x more than precision):** "
        f"precision={ev['metrics_chosen']['precision']:.4f}, "
        f"recall={ev['metrics_chosen']['recall']:.4f}, "
        f"f1={ev['metrics_chosen']['f1']:.4f} (validation F2={ev['f2_at_chosen']:.4f})",
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

    # Same split as GraphSAGE (splits.py)
    train_idx, val_idx, test_idx = split_indices(y.values)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    print(f"  Train: {len(X_train):,} rows ({y_train.mean()*100:.4f}% positive)")
    print(f"  Val:   {len(X_val):,} rows ({y_val.mean()*100:.4f}% positive)")
    print(f"  Test:  {len(X_test):,} rows ({y_test.mean()*100:.4f}% positive)")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight (train neg/pos): {scale_pos_weight:.2f}")

    # XGBoost
    print("Training XGBoost...")
    t0 = time.time()
    xgb_model = make_xgboost(scale_pos_weight)
    xgb_model.fit(X_train, y_train)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Random Forest
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

    # Evaluation
    print("Evaluating on test set...")
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    xgb_eval = evaluate_model("XGBoost", y_test, xgb_proba, y_val,
                              xgb_model.predict_proba(X_val)[:, 1])
    rf_eval = evaluate_model("Random Forest", y_test, rf_proba, y_val,
                             rf_model.predict_proba(X_val)[:, 1])

    print(f"XGBoost: AUC-ROC={xgb_eval['auc_roc']:.4f}, AUC-PR={xgb_eval['auc_pr']:.4f}")
    print(f"Random Forest: AUC-ROC={rf_eval['auc_roc']:.4f}, AUC-PR={rf_eval['auc_pr']:.4f}")

    # Feature importance
    xgb_importance = pd.Series(
        xgb_model.get_booster().get_score(importance_type="gain"), name="xgb_gain"
    )
    # Keys are feature names because X_train is a DataFrame
    xgb_importance = xgb_importance.reindex(feature_cols).fillna(0.0)
    xgb_importance = xgb_importance / xgb_importance.sum()  # normalize

    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols, name="rf_importance")

    importance_df = pd.DataFrame({"xgb_gain_normalized": xgb_importance, "rf_importance": rf_importance})
    importance_df["avg_rank"] = (
        importance_df["xgb_gain_normalized"].rank(ascending=False)
        + importance_df["rf_importance"].rank(ascending=False)
    ) / 2
    importance_df = importance_df.sort_values("avg_rank")

    # Save models
    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(XGB_MODEL_PATH))
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved models to {XGB_MODEL_PATH} and {RF_MODEL_PATH}")

    metrics = {
        ev["name"]: {
            "auc_roc": ev["auc_roc"], "auc_pr": ev["auc_pr"],
            "threshold": float(ev["threshold_chosen"]),
            **ev["metrics_chosen"],
            "confusion_matrix": ev["confusion_matrix_chosen"].tolist(),
        }
        for ev in (xgb_eval, rf_eval)
    }
    metrics["feature_importance_xgb_gain"] = xgb_importance.sort_values(ascending=False).to_dict()
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    # Report
    lines = [
        "# Baseline Model Summary",
        "",
        "## Setup",
        f"- Feature table: {df.shape[0]:,} rows, {len(feature_cols)} features "
        f"(excluded: {EXCLUDED_COLS[:-1]})",
        "- Split: shared 70/10/20 train/validation/test split from splits.py (stratified on "
        "`is_illicit`, random_state=42) — identical to the GraphSAGE split",
        f"- Train set: {len(X_train):,} rows ({y_train.mean()*100:.4f}% positive)",
        f"- Validation set: {len(X_val):,} rows ({y_val.mean()*100:.4f}% positive)",
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
        "maximizes the F2-score (recall weighted 4x more heavily than precision) on the "
        "VALIDATION set's precision-recall curve, and is then applied unchanged to the test set "
        "— the same protocol as GraphSAGE, so no test label influences any reported threshold. "
        "Both the default-0.5 and F2-optimal metrics are reported below.",
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
        "- Phase 6 flagged `community_illicit_density` and `cycle_count` as promising nonlinear "
        "signals (weak linear correlation but large mean gap between classes).",
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
