
"""
Experiment 10: Statistical Validation of Knowledge Inequality

Input:
    results_category_audit/raw_vs_kgsf_item_coverage.csv

Outputs:
    results_statistical_validation/

All analyses use the existing 6,924-item catalog.
No model retraining or catalog reconstruction.
"""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    kruskal,
    mannwhitneyu,
    wilcoxon,
    binomtest,
    rankdata,
)


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

INPUT = Path(
    "results_category_audit/raw_vs_kgsf_item_coverage.csv"
)

OUT = Path("results_statistical_validation")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261103
BOOTSTRAPS = 3000

rng = np.random.default_rng(SEED)


# ==========================================================
# 2. LOAD AND VALIDATE DATA
# ==========================================================

df = pd.read_csv(INPUT)

required = [
    "movie_entity_id",
    "frequency_group",
    "cold_start_type",
    "raw_semantic_connected",
    "raw_semantic_degree",
    "kgsf_semantic_connected",
    "kgsf_semantic_degree",
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["movie_entity_id"].duplicated().any():
    raise ValueError("Duplicate movie IDs detected.")

if len(df) != 6924:
    raise ValueError(
        f"Expected 6924 items, found {len(df)}"
    )

for col in [
    "raw_semantic_connected",
    "kgsf_semantic_connected",
]:
    df[col] = df[col].astype(bool)

print("=" * 65)
print("EXPERIMENT 10: STATISTICAL VALIDATION")
print("=" * 65)

print("Catalog items:", len(df))
print("Frequency groups:",
      sorted(df["frequency_group"].unique()))

print("Cold-start categories:")
print(df["cold_start_type"].value_counts().to_string())


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def cliffs_delta(x, y):
    """Positive values indicate larger values in x."""

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    u = mannwhitneyu(
        x, y, alternative="two-sided"
    ).statistic

    return float(2 * u / (len(x) * len(y)) - 1)


def cramers_v(table):
    chi2 = chi2_contingency(
        table, correction=False
    )[0]

    n = table.sum()
    denominator = min(table.shape) - 1

    if denominator <= 0 or n == 0:
        return np.nan

    return float(np.sqrt(chi2 / (n * denominator)))


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values."""

    p = np.asarray(p_values, dtype=float)

    if len(p) == 0:
        return p

    order = np.argsort(p)
    adjusted = np.empty(len(p))

    running_max = 0.0
    m = len(p)

    for rank, index in enumerate(order):
        value = (m - rank) * p[index]
        running_max = max(running_max, value)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    estimates = np.empty(BOOTSTRAPS)

    for i in range(BOOTSTRAPS):
        sample = rng.choice(
            values, size=len(values), replace=True
        )
        estimates[i] = sample.mean()

    return tuple(
        np.percentile(estimates, [2.5, 97.5])
    )


def bootstrap_proportion_ci(values):
    return bootstrap_mean_ci(
        np.asarray(values, dtype=float)
    )


# ==========================================================
# 4. CONNECTIVITY BY FREQUENCY GROUP
# ==========================================================

print("\n========== 1. CONNECTIVITY INEQUALITY ==========")

connectivity_results = []

for graph in ["raw", "kgsf"]:

    col = f"{graph}_semantic_connected"

    table = []

    for group in range(5):
        subset = df[df["frequency_group"] == group]

        connected = int(subset[col].sum())
        disconnected = len(subset) - connected

        table.append([connected, disconnected])

        ci_low, ci_high = bootstrap_proportion_ci(
            subset[col].to_numpy()
        )

        connectivity_results.append({
            "graph": graph.upper(),
            "group": f"G{group}",
            "items": len(subset),
            "connected": connected,
            "disconnected": disconnected,
            "coverage_pct": 100 * connected / len(subset),
            "ci95_lower_pct": 100 * ci_low,
            "ci95_upper_pct": 100 * ci_high,
        })

    table = np.asarray(table)

    chi2, p, dof, expected = chi2_contingency(
        table
    )

    v = cramers_v(table)

    print(f"\n{graph.upper()} GRAPH")
    print("Chi-square:", round(chi2, 4))
    print("Degrees of freedom:", dof)
    print("P-value:", p)
    print("Cramer's V:", round(v, 4))


connectivity_df = pd.DataFrame(connectivity_results)

connectivity_df.to_csv(
    OUT / "connectivity_by_group.csv",
    index=False
)


# ==========================================================
# 5. ZERO-SHOT VS ONE-SHOT
# ==========================================================

print("\n========== 2. ZERO-SHOT VS ONE-SHOT ==========")

cold_results = []

for graph in ["raw", "kgsf"]:

    col = f"{graph}_semantic_connected"

    zero = df[
        df["cold_start_type"].str.lower().str.contains(
            "zero", na=False
        )
    ]

    one = df[
        df["cold_start_type"].str.lower().str.contains(
            "one", na=False
        )
    ]

    if len(zero) == 0 or len(one) == 0:
        raise ValueError(
            "Could not identify zero-shot and one-shot groups."
        )

    table = np.array([
        [zero[col].sum(), len(zero) - zero[col].sum()],
        [one[col].sum(), len(one) - one[col].sum()],
    ])

    odds_ratio, p = fisher_exact(table)

    difference = (
        zero[col].mean() - one[col].mean()
    )

    ci_low, ci_high = bootstrap_mean_ci(
        zero[col].to_numpy().astype(float)
    )

    one_low, one_high = bootstrap_mean_ci(
        one[col].to_numpy().astype(float)
    )

    # Independent bootstrap of the difference.
    differences = np.empty(BOOTSTRAPS)

    z_values = zero[col].to_numpy().astype(float)
    o_values = one[col].to_numpy().astype(float)

    for i in range(BOOTSTRAPS):
        z_sample = rng.choice(
            z_values, len(z_values), replace=True
        )
        o_sample = rng.choice(
            o_values, len(o_values), replace=True
        )
        differences[i] = z_sample.mean() - o_sample.mean()

    diff_low, diff_high = np.percentile(
        differences, [2.5, 97.5]
    )

    result = {
        "graph": graph.upper(),
        "zero_shot_n": len(zero),
        "one_shot_n": len(one),
        "zero_shot_coverage_pct": 100 * zero[col].mean(),
        "one_shot_coverage_pct": 100 * one[col].mean(),
        "coverage_difference_pp": 100 * difference,
        "difference_ci95_lower_pp": 100 * diff_low,
        "difference_ci95_upper_pp": 100 * diff_high,
        "odds_ratio": odds_ratio,
        "fisher_p": p,
    }

    cold_results.append(result)

    print(f"\n{graph.upper()}")
    print("Zero-shot coverage:",
          round(100 * zero[col].mean(), 2), "%")
    print("One-shot coverage:",
          round(100 * one[col].mean(), 2), "%")
    print("Difference (percentage points):",
          round(100 * difference, 2))
    print("95% bootstrap CI:",
          (round(100 * diff_low, 2),
           round(100 * diff_high, 2)))
    print("Odds ratio:", odds_ratio)
    print("Fisher p-value:", p)


pd.DataFrame(cold_results).to_csv(
    OUT / "zero_vs_one_connectivity.csv",
    index=False
)


# ==========================================================
# 6. SEMANTIC DEGREE BY FREQUENCY GROUP
# ==========================================================

print("\n========== 3. SEMANTIC DEGREE ==========")

degree_results = []

for graph in ["raw", "kgsf"]:

    col = f"{graph}_semantic_degree"
    connected_col = f"{graph}_semantic_connected"

    groups_all = []
    groups_connected = []

    print(f"\n{graph.upper()} GRAPH")

    for group in range(5):

        subset = df[
            df["frequency_group"] == group
        ]

        all_values = subset[col].to_numpy()

        connected_values = subset.loc[
            subset[connected_col], col
        ].to_numpy()

        groups_all.append(all_values)
        groups_connected.append(connected_values)

        degree_results.append({
            "graph": graph.upper(),
            "group": f"G{group}",
            "items": len(all_values),
            "connected_items": len(connected_values),
            "mean_degree_all": all_values.mean(),
            "median_degree_all": np.median(all_values),
            "mean_degree_connected": (
                connected_values.mean()
                if len(connected_values) else np.nan
            ),
            "median_degree_connected": (
                np.median(connected_values)
                if len(connected_values) else np.nan
            ),
        })

    h_all, p_all = kruskal(*groups_all)

    h_connected, p_connected = kruskal(
        *groups_connected
    )

    print("All-item Kruskal-Wallis H:", h_all)
    print("All-item p-value:", p_all)

    print("Connected-only Kruskal-Wallis H:",
          h_connected)
    print("Connected-only p-value:",
          p_connected)


pd.DataFrame(degree_results).to_csv(
    OUT / "semantic_degree_by_group.csv",
    index=False
)


# ==========================================================
# 7. PAIRWISE G0-G4 COMPARISONS
# ==========================================================

print("\n========== 4. PAIRWISE COMPARISONS ==========")

pairwise_results = []

for graph in ["raw", "kgsf"]:

    for population in ["all_items", "connected_only"]:

        col = f"{graph}_semantic_degree"
        connected_col = f"{graph}_semantic_connected"

        comparisons = []

        for g1, g2 in itertools.combinations(range(5), 2):

            x = df.loc[
                df["frequency_group"] == g1, col
            ]

            y = df.loc[
                df["frequency_group"] == g2, col
            ]

            if population == "connected_only":
                x = x[df.loc[x.index, connected_col]]
                y = y[df.loc[y.index, connected_col]]

            if len(x) == 0 or len(y) == 0:
                continue

            test = mannwhitneyu(
                x, y, alternative="two-sided"
            )

            comparisons.append({
                "graph": graph.upper(),
                "population": population,
                "group_1": f"G{g1}",
                "group_2": f"G{g2}",
                "n_1": len(x),
                "n_2": len(y),
                "median_1": x.median(),
                "median_2": y.median(),
                "mann_whitney_u": test.statistic,
                "p_raw": test.pvalue,
                "cliffs_delta": cliffs_delta(x, y),
            })

        adjusted = holm_adjust([
            item["p_raw"] for item in comparisons
        ])

        for item, p_adj in zip(comparisons, adjusted):
            item["p_holm"] = p_adj
            item["significant_005"] = p_adj < 0.05
            pairwise_results.append(item)

pairwise_df = pd.DataFrame(pairwise_results)

pairwise_df.to_csv(
    OUT / "pairwise_degree_comparisons.csv",
    index=False
)

print("Pairwise comparisons completed:",
      len(pairwise_df))


# ==========================================================
# 8. ZERO-SHOT VS ONE-SHOT DEGREE
# ==========================================================

print("\n========== 5. COLD-START DEGREE ==========")

cold_degree_results = []

for graph in ["raw", "kgsf"]:

    col = f"{graph}_semantic_degree"

    for population in ["all_items", "connected_only"]:

        zero = df[
            df["cold_start_type"].str.lower().str.contains(
                "zero", na=False
            )
        ]

        one = df[
            df["cold_start_type"].str.lower().str.contains(
                "one", na=False
            )
        ]

        if population == "connected_only":
            zero = zero[zero[f"{graph}_semantic_connected"]]
            one = one[one[f"{graph}_semantic_connected"]]

        x = zero[col].to_numpy()
        y = one[col].to_numpy()

        test = mannwhitneyu(
            x, y, alternative="two-sided"
        )

        cold_degree_results.append({
            "graph": graph.upper(),
            "population": population,
            "zero_shot_n": len(x),
            "one_shot_n": len(y),
            "zero_shot_median": np.median(x),
            "one_shot_median": np.median(y),
            "zero_shot_mean": np.mean(x),
            "one_shot_mean": np.mean(y),
            "mann_whitney_u": test.statistic,
            "p_value": test.pvalue,
            "cliffs_delta": cliffs_delta(x, y),
        })

        print(
            graph.upper(),
            population,
            "| p =", test.pvalue,
            "| Cliff's delta =", cliffs_delta(x, y)
        )

pd.DataFrame(cold_degree_results).to_csv(
    OUT / "zero_vs_one_degree.csv",
    index=False
)


# ==========================================================
# 9. RAW VS KGSF PAIRED COMPARISON
# ==========================================================

print("\n========== 6. RAW VS KGSF RETENTION ==========")

raw = df["raw_semantic_connected"]
kgsf = df["kgsf_semantic_connected"]

raw_only = int((raw & ~kgsf).sum())
kgsf_only = int((~raw & kgsf).sum())

discordant = raw_only + kgsf_only

mcnemar_p = (
    binomtest(
        min(raw_only, kgsf_only),
        discordant,
        p=0.5
    ).pvalue
    if discordant > 0 else 1.0
)

print("Raw-only connected:", raw_only)
print("KGSF-only connected:", kgsf_only)
print("Exact McNemar p-value:", mcnemar_p)

raw_degree = df["raw_semantic_degree"]
kgsf_degree = df["kgsf_semantic_degree"]

degree_difference = raw_degree - kgsf_degree

if np.all(degree_difference == 0):
    wilcoxon_stat = 0.0
    wilcoxon_p = 1.0
else:
    test = wilcoxon(
        raw_degree,
        kgsf_degree,
        alternative="two-sided",
        zero_method="wilcox"
    )
    wilcoxon_stat = test.statistic
    wilcoxon_p = test.pvalue

print("Mean raw degree:", raw_degree.mean())
print("Mean KGSF degree:", kgsf_degree.mean())
print("Mean degree reduction:", degree_difference.mean())
print("Wilcoxon statistic:", wilcoxon_stat)
print("Wilcoxon p-value:", wilcoxon_p)

pd.DataFrame([{
    "raw_only_connected": raw_only,
    "kgsf_only_connected": kgsf_only,
    "mcnemar_exact_p": mcnemar_p,
    "mean_raw_degree": raw_degree.mean(),
    "mean_kgsf_degree": kgsf_degree.mean(),
    "mean_degree_difference": degree_difference.mean(),
    "wilcoxon_statistic": wilcoxon_stat,
    "wilcoxon_p": wilcoxon_p,
}]).to_csv(
    OUT / "raw_vs_kgsf_paired_tests.csv",
    index=False
)


# ==========================================================
# 10. SAVE STATISTICAL SUMMARY
# ==========================================================

summary = {
    "experiment": 10,
    "catalog_items": len(df),
    "frequency_groups": 5,
    "bootstrap_iterations": BOOTSTRAPS,
    "random_seed": SEED,
    "zero_shot_items": int(
        df["cold_start_type"].str.lower().str.contains(
            "zero", na=False
        ).sum()
    ),
    "one_shot_items": int(
        df["cold_start_type"].str.lower().str.contains(
            "one", na=False
        ).sum()
    ),
    "raw_connected_items": int(raw.sum()),
    "kgsf_connected_items": int(kgsf.sum()),
    "raw_only_connected": raw_only,
    "kgsf_only_connected": kgsf_only,
    "mcnemar_exact_p": float(mcnemar_p),
    "wilcoxon_degree_p": float(wilcoxon_p),
    "interpretation_warning": (
        "Statistical significance does not establish causality. "
        "Item-level bootstrap intervals quantify uncertainty "
        "under item resampling, not conversational sampling."
    ),
}

with open(
    OUT / "statistical_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)
print("EXPERIMENT 10 COMPLETED")
print("=" * 65)

print("Output directory:", OUT.resolve())

for path in sorted(OUT.iterdir()):
    if path.is_file():
        print(path.name, "|", path.stat().st_size, "bytes")
