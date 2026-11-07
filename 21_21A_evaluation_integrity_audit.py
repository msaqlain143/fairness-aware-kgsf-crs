
from pathlib import Path
import json
import pandas as pd
import numpy as np

# ============================================================
# EXPERIMENT 21.21A
# Evaluation Integrity and Baseline Reconciliation
# ============================================================

ROOT = Path(".")
INPUT = (
    ROOT / "results_corrected" /
    "m2_validation_fairness_accuracy_tradeoff.csv"
)

OUT = ROOT / "results_corrected" / "experiment_21_21A"
OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input: {INPUT}")

df = pd.read_csv(INPUT)

required = ["method", "split"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("=" * 75)
print("EXPERIMENT 21.21A: EVALUATION INTEGRITY AUDIT")
print("=" * 75)

# ------------------------------------------------------------
# 1. SOURCE INVENTORY
# ------------------------------------------------------------

print("\n[1] SOURCE INVENTORY")

print("Input:", INPUT)
print("Rows:", len(df))
print("Methods:", df["method"].nunique())
print("Splits:", df["split"].nunique())

print("\nMethod distribution:")
print(df["method"].value_counts(dropna=False))

print("\nSplit distribution:")
print(df["split"].value_counts(dropna=False))

# ------------------------------------------------------------
# 2. DUPLICATE CHECK
# ------------------------------------------------------------

print("\n[2] DUPLICATE CHECK")

duplicate_keys = df[
    df.duplicated(["method", "split"], keep=False)
].copy()

print("Duplicate method/split records:", len(duplicate_keys))

duplicate_keys.to_csv(
    OUT / "duplicate_method_split_records.csv",
    index=False
)

# ------------------------------------------------------------
# 3. POINT ESTIMATE VS CONFIDENCE INTERVAL
# ------------------------------------------------------------

print("\n[3] CONFIDENCE INTERVAL INTEGRITY")

metric_columns = [
    c for c in df.columns
    if not c.endswith("_ci95_low")
    and not c.endswith("_ci95_high")
    and not c.endswith("_delta_vs_baseline")
    and not c.endswith("_delta_ci95_low")
    and not c.endswith("_delta_ci95_high")
]

ci_records = []

for metric in metric_columns:

    low_col = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    if metric not in df.columns:
        continue

    if low_col not in df.columns or high_col not in df.columns:
        continue

    for idx, row in df.iterrows():

        point = pd.to_numeric(
            pd.Series([row[metric]]), errors="coerce"
        ).iloc[0]

        low = pd.to_numeric(
            pd.Series([row[low_col]]), errors="coerce"
        ).iloc[0]

        high = pd.to_numeric(
            pd.Series([row[high_col]]), errors="coerce"
        ).iloc[0]

        if pd.isna(point) or pd.isna(low) or pd.isna(high):
            continue

        if low > high:
            status = "INVALID_INTERVAL_ORDER"

        elif point < low or point > high:
            status = "POINT_OUTSIDE_INTERVAL"

        else:
            status = "PASS"

        ci_records.append({
            "method": row["method"],
            "split": row["split"],
            "metric": metric,
            "point_estimate": float(point),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "status": status
        })

ci_df = pd.DataFrame(ci_records)

if not ci_df.empty:

    print(ci_df["status"].value_counts())

    print("\nFlagged intervals:")

    print(
        ci_df[ci_df["status"] != "PASS"].to_string(index=False)
    )

    ci_df.to_csv(
        OUT / "point_estimate_ci_integrity.csv",
        index=False
    )

# ------------------------------------------------------------
# 4. DELTA VS BASELINE INTERVAL CHECK
# ------------------------------------------------------------

print("\n[4] DELTA CONFIDENCE INTERVAL CHECK")

delta_records = []

for metric in df.columns:

    if not metric.endswith("_delta_vs_baseline"):
        continue

    prefix = metric[:-len("_delta_vs_baseline")]

    low_col = f"{prefix}_delta_ci95_low"
    high_col = f"{prefix}_delta_ci95_high"

    if low_col not in df.columns or high_col not in df.columns:
        continue

    for _, row in df.iterrows():

        point, low, high = [
            pd.to_numeric(
                pd.Series([row[c]]), errors="coerce"
            ).iloc[0]
            for c in [metric, low_col, high_col]
        ]

        if pd.isna(point) or pd.isna(low) or pd.isna(high):
            continue

        if low > high:
            status = "INVALID_INTERVAL_ORDER"
        elif point < low or point > high:
            status = "POINT_OUTSIDE_INTERVAL"
        else:
            status = "PASS"

        delta_records.append({
            "method": row["method"],
            "split": row["split"],
            "metric": prefix,
            "delta": float(point),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "status": status
        })

delta_df = pd.DataFrame(delta_records)

if not delta_df.empty:

    print(delta_df["status"].value_counts())

    print("\nFlagged delta intervals:")

    print(
        delta_df[delta_df["status"] != "PASS"]
        .to_string(index=False)
    )

    delta_df.to_csv(
        OUT / "delta_ci_integrity.csv",
        index=False
    )

# ------------------------------------------------------------
# 5. BASELINE AND METHOD COMPARISON
# ------------------------------------------------------------

print("\n[5] METHOD COMPARISON")

comparison_metrics = [
    c for c in [
        "hit10",
        "hit50",
        "ndcg10",
        "catalog_coverage10",
        "item_gini",
        "group_0_exposure_share",
        "group_1_exposure_share",
        "group_2_exposure_share",
        "group_3_exposure_share",
        "group_4_exposure_share"
    ]
    if c in df.columns
]

comparison = df[
    ["method", "split"] + comparison_metrics
].copy()

print(comparison.to_string(index=False))

comparison.to_csv(
    OUT / "method_metric_comparison.csv",
    index=False
)

# ------------------------------------------------------------
# 6. BASELINE ROW IDENTIFICATION
# ------------------------------------------------------------

print("\n[6] BASELINE IDENTIFICATION")

baseline_candidates = df[
    df["method"].astype(str).str.contains(
        "baseline|a0", case=False, na=False
    )
].copy()

print("Candidate baseline rows:", len(baseline_candidates))

print(
    baseline_candidates[
        ["method", "split"] + comparison_metrics
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 7. FINAL VALIDATION
# ------------------------------------------------------------

summary = {
    "experiment": "21.21A",
    "input_file": str(INPUT),
    "total_rows": int(len(df)),
    "unique_methods": int(df["method"].nunique()),
    "unique_splits": int(df["split"].nunique()),
    "duplicate_method_split_rows": int(len(duplicate_keys)),
    "confidence_interval_checks": int(len(ci_df)),
    "confidence_interval_failures": int(
        (ci_df["status"] != "PASS").sum()
    ) if not ci_df.empty else 0,
    "delta_interval_checks": int(len(delta_df)),
    "delta_interval_failures": int(
        (delta_df["status"] != "PASS").sum()
    ) if not delta_df.empty else 0,
    "baseline_candidate_rows": int(len(baseline_candidates)),
    "original_results_modified": False
}

with open(
    OUT / "experiment_21_21A_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

print("\n[7] FINAL SUMMARY")

print(json.dumps(summary, indent=4))

print("\nOUTPUT DIRECTORY:", OUT)
print("\nEXPERIMENT 21.21A COMPLETED")
