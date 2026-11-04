
import csv
import json
import pickle
from pathlib import Path

import numpy as np


# ============================================================
# EXPERIMENT 17
# RANDOMIZED TIE-BREAKING SENSITIVITY ANALYSIS
# ============================================================

CATALOG_FILE = Path("catalog_audit_corrected.pkl")

EVENTS_FILE = Path(
    "results_final_test_m2/"
    "adaptive_m2_with_kgsf_baseline_events.csv"
)

OUT = Path("results_tie_breaking_sensitivity")

METHODS = [
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_1",
    "A2_frequency_context_1",
    "A1_frequency_3",
    "A2_frequency_context_3",
]

B = 1000
SEED = 20261104

EXPECTED_EVENTS = 4003
EXPECTED_CATALOG = 6924
TOP_K = 10

rng = np.random.default_rng(SEED)

OUT.mkdir(exist_ok=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_csv(path, rows):
    require(len(rows) > 0, f"No data to save: {path}")

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


print("=" * 72)
print("EXPERIMENT 17: TIE-BREAKING SENSITIVITY ANALYSIS")
print("=" * 72)


# ============================================================
# 1. LOAD CANONICAL CATALOG
# ============================================================

with CATALOG_FILE.open("rb") as f:
    catalog = pickle.load(f)

movie_ids = np.array(
    sorted(map(int, catalog["movie_ids"])),
    dtype=np.int64
)

freq_dict = {
    int(k): int(v)
    for k, v in catalog["freq_dict"].items()
}

original_groups = {
    int(k): int(v)
    for k, v in catalog["item2group"].items()
}

N = len(movie_ids)

require(N == EXPECTED_CATALOG, "Catalog size mismatch.")
require(set(freq_dict) == set(movie_ids), "Frequency keys mismatch.")
require(set(original_groups) == set(movie_ids), "Group keys mismatch.")

frequencies = np.array(
    [freq_dict[int(x)] for x in movie_ids],
    dtype=np.int64
)

original_assignment = np.array(
    [original_groups[int(x)] for x in movie_ids],
    dtype=np.int64
)

require(
    set(original_assignment) == set(range(5)),
    "Invalid original group labels."
)

original_sizes = np.bincount(
    original_assignment,
    minlength=5
)

print("\n1. CATALOG")

print("Catalog:", N)
print("Original group sizes:", original_sizes.tolist())
print("Replicates:", B)


# ============================================================
# 2. LOAD FROZEN RECOMMENDATIONS
# ============================================================

print("\n2. RECOMMENDATION DATA")

catalog_index = {
    int(movie): i
    for i, movie in enumerate(movie_ids)
}

exposure = {
    method: np.zeros(N, dtype=np.int64)
    for method in METHODS
}

event_counts = {
    method: 0
    for method in METHODS
}

seen_keys = set()

with EVENTS_FILE.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        method = row["method"].strip()

        require(method in METHODS, f"Unknown method: {method}")

        event_id = int(row["event_id"])

        key = (event_id, method)

        require(key not in seen_keys, f"Duplicate record: {key}")

        seen_keys.add(key)

        movies = json.loads(row["top10_movie_ids"])

        require(
            len(movies) == TOP_K,
            f"Invalid recommendation length: {key}"
        )

        movies = list(map(int, movies))

        require(
            len(set(movies)) == TOP_K,
            f"Duplicate recommendation: {key}"
        )

        require(
            all(x in catalog_index for x in movies),
            f"Unknown recommended item: {key}"
        )

        indices = np.array(
            [catalog_index[x] for x in movies],
            dtype=np.int64
        )

        exposure[method][indices] += 1

        event_counts[method] += 1

expected_records = EXPECTED_EVENTS * len(METHODS)

require(
    len(seen_keys) == expected_records,
    "Incomplete event-method records."
)

for method in METHODS:

    require(
        event_counts[method] == EXPECTED_EVENTS,
        f"Event count mismatch: {method}"
    )

    require(
        exposure[method].sum() == EXPECTED_EVENTS * TOP_K,
        f"Exposure conservation failed: {method}"
    )

print("Event-method records:", len(seen_keys))
print("Recommendation slots per method:", EXPECTED_EVENTS * TOP_K)
print("Recommendation integrity: PASS")


# ============================================================
# 3. ORIGINAL GROUPING REFERENCE
# ============================================================

print("\n3. ORIGINAL GROUPING REFERENCE")

original_results = {}

for method in METHODS:

    values = exposure[method]

    group_exposure = np.bincount(
        original_assignment,
        weights=values,
        minlength=5
    )

    group_unique = np.bincount(
        original_assignment[values > 0],
        minlength=5
    )

    total = values.sum()

    original_results[method] = {
        "exposure_share": group_exposure / total,
        "item_coverage": group_unique / original_sizes,
    }

    print(
        f"{method:25s}",
        "Group exposure:",
        np.round(group_exposure / total, 5)
    )


# ============================================================
# 4. RANDOMIZED TIE-BREAKING
# ============================================================

print("\n4. RANDOMIZED TIE-BREAKING")

exposure_samples = {
    method: np.zeros((B, 5))
    for method in METHODS
}

coverage_samples = {
    method: np.zeros((B, 5))
    for method in METHODS
}

for b in range(B):

    # Random priority is assigned to every item.
    # Sorting by frequency first ensures that only
    # equal-frequency items change their relative order.

    random_priority = rng.permutation(N)

    ordering = np.lexsort(
        (random_priority, frequencies)
    )

    randomized_items = movie_ids[ordering]

    randomized_groups = np.empty(N, dtype=np.int64)

    split_groups = np.array_split(
        randomized_items,
        5
    )

    for group_id, items in enumerate(split_groups):

        indices = np.array(
            [catalog_index[int(x)] for x in items],
            dtype=np.int64
        )

        randomized_groups[indices] = group_id

    # Preserve original quintile sizes.
    randomized_sizes = np.bincount(
        randomized_groups,
        minlength=5
    )

    require(
        np.array_equal(randomized_sizes, original_sizes),
        "Randomized group sizes changed."
    )

    for method in METHODS:

        values = exposure[method]

        total = values.sum()

        group_exposure = np.bincount(
            randomized_groups,
            weights=values,
            minlength=5
        )

        group_unique = np.bincount(
            randomized_groups[values > 0],
            minlength=5
        )

        exposure_samples[method][b] = (
            group_exposure / total
        )

        coverage_samples[method][b] = (
            group_unique / randomized_sizes
        )

    if (b + 1) % 250 == 0:
        print(f"Completed {b + 1}/{B}")


# ============================================================
# 5. SENSITIVITY SUMMARY
# ============================================================

print("\n5. SENSITIVITY SUMMARY")

summary_rows = []

for method in METHODS:

    for group in range(5):

        original_share = float(
            original_results[method]["exposure_share"][group]
        )

        original_coverage = float(
            original_results[method]["item_coverage"][group]
        )

        exposure_values = exposure_samples[method][:, group]

        coverage_values = coverage_samples[method][:, group]

        summary_rows.append({

            "method": method,
            "group": group,

            "original_exposure_share": original_share,

            "random_mean_exposure_share": float(
                np.mean(exposure_values)
            ),

            "exposure_q025": float(
                np.quantile(exposure_values, 0.025)
            ),

            "exposure_q975": float(
                np.quantile(exposure_values, 0.975)
            ),

            "original_item_coverage": original_coverage,

            "random_mean_item_coverage": float(
                np.mean(coverage_values)
            ),

            "coverage_q025": float(
                np.quantile(coverage_values, 0.025)
            ),

            "coverage_q975": float(
                np.quantile(coverage_values, 0.975)
            ),

            "original_exposure_within_range": bool(
                np.quantile(exposure_values, 0.025)
                <= original_share
                <= np.quantile(exposure_values, 0.975)
            ),

            "original_coverage_within_range": bool(
                np.quantile(coverage_values, 0.025)
                <= original_coverage
                <= np.quantile(coverage_values, 0.975)
            ),

        })


# ============================================================
# 6. SAVE RANDOMIZATION DISTRIBUTIONS
# ============================================================

print("\n6. SAVING RESULTS")

save_csv(
    OUT / "tie_breaking_sensitivity_summary.csv",
    summary_rows
)

np.savez_compressed(
    OUT / "tie_breaking_randomization_distributions.npz",
    **{
        f"exposure_{method}": exposure_samples[method]
        for method in METHODS
    },
    **{
        f"coverage_{method}": coverage_samples[method]
        for method in METHODS
    }
)

# ============================================================
# 7. VALIDATION SUMMARY
# ============================================================

validation = {

    "experiment": 17,

    "experiment_name": (
        "Randomized Tie-Breaking Sensitivity Analysis"
    ),

    "catalog_size": N,

    "events": EXPECTED_EVENTS,

    "methods": METHODS,

    "randomization_replicates": B,

    "seed": SEED,

    "original_group_sizes": original_sizes.tolist(),

    "validation": {

        "catalog_integrity": "PASS",

        "recommendation_integrity": "PASS",

        "event_method_completeness": "PASS",

        "exposure_conservation": "PASS",

        "group_size_preservation": "PASS",

        "randomized_tie_breaking": "PASS",

    },

    "interpretation": (

        "Reported quantiles describe sensitivity to randomized "
        "tie-breaking under fixed frequency values and fixed "
        "recommendation lists. They are not confidence intervals "
        "for population-level recommendation performance."
    ),

    "outputs": [

        "tie_breaking_sensitivity_summary.csv",

        "tie_breaking_randomization_distributions.npz",

    ],

}

with (
    OUT / "tie_breaking_sensitivity_summary.json"
).open("w", encoding="utf-8") as f:

    json.dump(validation, f, indent=2)


print("\n" + "=" * 72)
print("EXPERIMENT 17 COMPLETED")
print("=" * 72)

print("Catalog:", N)
print("Events:", EXPECTED_EVENTS)
print("Methods:", len(METHODS))
print("Randomizations:", B)

print("Recommendation integrity: PASS")
print("Group-size preservation: PASS")
print("Exposure sensitivity analysis: PASS")

print("Output:", OUT.resolve())

print("\nNo model training performed.")
print("No recommendation inference performed.")
print("Frozen test recommendations were not modified.")
