
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# EXPERIMENT 18
# PAIRED TIE-BREAKING ROBUSTNESS ANALYSIS
# ============================================================

ROOT = Path("results_tie_breaking_sensitivity")

ARCHIVE = ROOT / "tie_breaking_randomization_distributions.npz"

REFERENCE = ROOT / "tie_breaking_sensitivity_summary.csv"

OUT = Path("results_paired_tie_robustness")

METHODS = [
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_1",
    "A2_frequency_context_1",
    "A1_frequency_3",
    "A2_frequency_context_3",
]

COMPARISONS = [
    ("A1_frequency_3", "KGSF_only"),
    ("A1_frequency_3", "A0_fixed_025"),
    ("A1_frequency_3", "A1_frequency_1"),
    ("A2_frequency_context_3", "A1_frequency_3"),
]

METRICS = ["exposure", "coverage"]

OUT.mkdir(exist_ok=False)

print("=" * 75)
print("EXPERIMENT 18: PAIRED TIE-BREAKING ROBUSTNESS")
print("=" * 75)


# ============================================================
# 1. LOAD RANDOMIZATION DISTRIBUTIONS
# ============================================================

archive = np.load(ARCHIVE, allow_pickle=False)

reference = pd.read_csv(REFERENCE)

print("\n1. INPUT VALIDATION")

distributions = {}

for method in METHODS:

    for metric in METRICS:

        key = f"{metric}_{method}"

        if key not in archive.files:
            raise RuntimeError(f"Missing archive array: {key}")

        values = archive[key]

        if values.ndim != 2 or values.shape[1] != 5:
            raise RuntimeError(f"Invalid array shape: {key}")

        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values: {key}")

        distributions[(method, metric)] = values

B = distributions[(METHODS[0], "exposure")].shape[0]

for values in distributions.values():

    if values.shape[0] != B:
        raise RuntimeError("Bootstrap/randomization counts differ.")

print("Methods:", len(METHODS))
print("Randomizations:", B)
print("Metrics:", METRICS)


# ============================================================
# 2. RECONSTRUCT ORIGINAL RESULTS
# ============================================================

original = {}

for _, row in reference.iterrows():

    key = (row["method"], int(row["group"]))

    original[key] = {
        "exposure": float(row["original_exposure_share"]),
        "coverage": float(row["original_item_coverage"]),
    }

expected = {
    (method, group)
    for method in METHODS
    for group in range(5)
}

if set(original) != expected:
    raise RuntimeError("Original reference records incomplete.")

print("\n2. ORIGINAL RESULTS")
print("Reference records:", len(original))


# ============================================================
# 3. PAIRED DIFFERENCES
# ============================================================

rows = []

print("\n3. PAIRED COMPARISONS")

for proposed, baseline in COMPARISONS:

    for metric in METRICS:

        proposed_values = distributions[(proposed, metric)]

        baseline_values = distributions[(baseline, metric)]

        # Same randomized assignment is used for both methods.
        # Therefore, compare replicate-by-replicate differences.

        deltas = proposed_values - baseline_values

        if not np.isfinite(deltas).all():
            raise RuntimeError("Invalid paired differences.")

        for group in range(5):

            values = deltas[:, group]

            original_delta = (
                original[(proposed, group)][metric]
                - original[(baseline, group)][metric]
            )

            lower = float(np.quantile(values, 0.025))
            upper = float(np.quantile(values, 0.975))

            row = {

                "proposed": proposed,

                "baseline": baseline,

                "metric": metric,

                "group": group,

                "original_difference": float(original_delta),

                "randomization_mean_difference": float(
                    np.mean(values)
                ),

                "randomization_sd": float(
                    np.std(values, ddof=1)
                ),

                "difference_q025": lower,

                "difference_median": float(
                    np.median(values)
                ),

                "difference_q975": upper,

                "fraction_positive": float(
                    np.mean(values > 0)
                ),

                "fraction_negative": float(
                    np.mean(values < 0)
                ),

                "original_difference_within_range": bool(
                    lower <= original_delta <= upper
                ),

            }

            rows.append(row)

            print(
                f"{proposed} vs {baseline} | "
                f"{metric} | G{group} | "
                f"Original={original_delta:+.5f} | "
                f"Random mean={np.mean(values):+.5f} | "
                f"Range=[{lower:+.5f}, {upper:+.5f}]"
            )


# ============================================================
# 4. SAVE RESULTS
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUT / "paired_tie_breaking_comparisons.csv",
    index=False
)

# Summary by comparison and metric.
summary_rows = []

for (proposed, baseline), group_df in df.groupby(
    ["proposed", "baseline"]
):

    for metric in METRICS:

        selected = group_df[
            group_df["metric"] == metric
        ]

        summary_rows.append({

            "proposed": proposed,

            "baseline": baseline,

            "metric": metric,

            "groups_with_positive_mean_difference": int(
                (
                    selected["randomization_mean_difference"] > 0
                ).sum()
            ),

            "groups_with_original_difference_outside_range": int(
                (
                    ~selected["original_difference_within_range"]
                ).sum()
            ),

            "mean_fraction_positive_across_groups": float(
                selected["fraction_positive"].mean()
            ),

        })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUT / "paired_tie_breaking_summary.csv",
    index=False
)


# ============================================================
# 5. VALIDATION
# ============================================================

if len(df) != len(COMPARISONS) * len(METRICS) * 5:
    raise RuntimeError("Unexpected comparison row count.")

if not df["original_difference_within_range"].notna().all():
    raise RuntimeError("Missing sensitivity checks.")

metadata = {

    "experiment": 18,

    "name": "Paired Tie-Breaking Robustness Analysis",

    "randomizations": B,

    "methods": METHODS,

    "comparisons": [
        {
            "proposed": p,
            "baseline": b
        }
        for p, b in COMPARISONS
    ],

    "metrics": METRICS,

    "groups": 5,

    "comparison_rows": len(df),

    "interpretation": (
        "Paired randomization sensitivity analysis. "
        "Percentile ranges describe tie-breaking variability, "
        "not population confidence intervals or hypothesis-test "
        "p-values."
    ),

    "validation": "PASS",

}

with (
    OUT / "paired_tie_breaking_metadata.json"
).open("w", encoding="utf-8") as f:

    json.dump(metadata, f, indent=2)


print("\n" + "=" * 75)
print("EXPERIMENT 18 COMPLETED")
print("=" * 75)

print("Paired comparison rows:", len(df))
print("Comparison pairs:", len(COMPARISONS))
print("Metrics:", len(METRICS))
print("Randomizations:", B)
print("Validation: PASS")
print("Output:", OUT.resolve())
