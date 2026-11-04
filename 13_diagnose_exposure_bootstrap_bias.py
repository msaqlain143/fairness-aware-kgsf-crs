
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("results_bootstrap_validation")

print("=" * 70)
print("EXPERIMENT 13: EXPOSURE BOOTSTRAP BIAS DIAGNOSTICS")
print("=" * 70)

# --------------------------------------------------
# 1. Load validation summary
# --------------------------------------------------

summary_path = ROOT / "bootstrap_validation_summary.json"

with summary_path.open(encoding="utf-8") as f:
    summary = json.load(f)

print("\n========== 1. SUMMARY STRUCTURE ==========")

print("Top-level keys:", list(summary.keys()))

# --------------------------------------------------
# 2. Inspect bootstrap archive
# --------------------------------------------------

archive_path = ROOT / "bootstrap_replicates.npz"

archive = np.load(archive_path, allow_pickle=False)

print("\n========== 2. BOOTSTRAP ARCHIVE ==========")

print("Archive keys:")

for key in archive.files:
    arr = archive[key]

    print(
        f"{key:40s}",
        "shape:", arr.shape,
        "dtype:", arr.dtype
    )

# --------------------------------------------------
# 3. Inspect interval diagnostics
# --------------------------------------------------

interval_path = ROOT / "bootstrap_interval_diagnostics.csv"

intervals = pd.read_csv(interval_path)

print("\n========== 3. INTERVAL DIAGNOSTICS ==========")

print("Columns:", intervals.columns.tolist())
print("Rows:", len(intervals))

print(intervals.head(15).to_string(index=False))

# --------------------------------------------------
# 4. Inspect distribution diagnostics
# --------------------------------------------------

distribution_path = ROOT / "bootstrap_distribution_diagnostics.csv"

distributions = pd.read_csv(distribution_path)

print("\n========== 4. DISTRIBUTION DIAGNOSTICS ==========")

print("Columns:", distributions.columns.tolist())
print("Rows:", len(distributions))

print(distributions.head(15).to_string(index=False))

# --------------------------------------------------
# 5. Inspect paired comparisons
# --------------------------------------------------

comparison_path = ROOT / "bootstrap_paired_comparisons.csv"

comparisons = pd.read_csv(comparison_path)

print("\n========== 5. PAIRED COMPARISONS ==========")

print("Columns:", comparisons.columns.tolist())
print("Rows:", len(comparisons))

print(comparisons.head(15).to_string(index=False))

# --------------------------------------------------
# 6. Identify coverage and Gini records
# --------------------------------------------------

print("\n========== 6. COVERAGE AND GINI ==========")

for metric in ["coverage10", "item_gini"]:

    selected = intervals[
        intervals["metric"] == metric
    ]

    print(f"\n--- {metric} ---")

    if selected.empty:
        print("No matching interval records.")
    else:
        print(selected.to_string(index=False))

# --------------------------------------------------
# 7. Identify paired fairness comparisons
# --------------------------------------------------

print("\n========== 7. FAIRNESS COMPARISONS ==========")

selected = comparisons[
    comparisons["metric"].isin(
        ["coverage10", "item_gini"]
    )
]

print(selected.to_string(index=False))

# --------------------------------------------------
# 8. Final diagnostic
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 13 PREPARATION COMPLETE")
print("=" * 70)

print("No model inference performed.")
print("No existing results modified.")
print("No bootstrap experiments repeated.")
