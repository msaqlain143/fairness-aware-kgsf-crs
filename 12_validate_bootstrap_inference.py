

"""
EXPERIMENT 12: BOOTSTRAP INFERENCE VALIDATION

Purpose:
1. Independently reproduce Experiment 11.
2. Validate data integrity and method pairing.
3. Diagnose bootstrap bias and dispersion.
4. Compare percentile and basic bootstrap intervals.
5. Investigate coverage and Gini behavior.
6. Recalculate paired comparisons and Holm corrections.

No model training or inference is performed.
Existing Experiment 11 outputs are never overwritten.
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path("results_final_test_m2")
CATALOG_FILE = Path("catalog_audit_corrected.pkl")

EVENTS_FILE = ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"
MAPPING_FILE = ROOT / "m2_test_event_conversation_mapping.csv"

PREVIOUS_RESULTS = Path(
    "results_cluster_bootstrap/cluster_bootstrap_results.csv"
)
PREVIOUS_COMPARISONS = Path(
    "results_cluster_bootstrap/cluster_bootstrap_comparisons.csv"
)

OUT = Path("results_bootstrap_validation")

METHODS = [
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_1",
    "A2_frequency_context_1",
    "A1_frequency_3",
    "A2_frequency_context_3",
]

COMPARISONS = [
    ("A0_fixed_025", "KGSF_only"),
    ("A1_frequency_1", "KGSF_only"),
    ("A1_frequency_3", "KGSF_only"),
    ("A2_frequency_context_3", "A1_frequency_3"),
]

B = 2000
SEED = 20261103
ALPHA = 0.05

EXPECTED_CATALOG = 6924
EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283

rng = np.random.default_rng(SEED)

OUT.mkdir(exist_ok=True)

OUTPUT_FILES = [
    "bootstrap_distribution_diagnostics.csv",
    "bootstrap_interval_diagnostics.csv",
    "bootstrap_paired_comparisons.csv",
    "bootstrap_validation_summary.json",
    "bootstrap_replicates.npz",
]

for filename in OUTPUT_FILES:
    if (OUT / filename).exists():
        raise FileExistsError(
            f"Output already exists: {OUT / filename}. "
            "Use a new output directory or archive previous results."
        )


# ============================================================
# 2. UTILITIES
# ============================================================

def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No records to save: {path}")

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


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)

    m = len(pvalues)
    order = np.argsort(pvalues)

    adjusted = np.empty(m)
    running = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running

    return adjusted


def normalized_value(value):
    """Normalize numeric metadata for cross-method comparisons."""

    if value is None or str(value).strip() == "":
        return None

    try:
        number = float(value)

        if np.isnan(number):
            return None

        return number

    except (ValueError, TypeError):
        return str(value).strip()


def percentile_interval(values):
    return np.percentile(values, [2.5, 97.5])


# ============================================================
# 3. LOAD AND VALIDATE CATALOG
# ============================================================

print("\n========== 1. CATALOG VALIDATION ==========")

with CATALOG_FILE.open("rb") as f:
    catalog = pickle.load(f)

catalog_ids = list(map(int, catalog["movie_ids"]))
catalog_set = set(catalog_ids)

if len(catalog_ids) != EXPECTED_CATALOG:
    raise RuntimeError("Unexpected catalog size.")

if len(catalog_set) != EXPECTED_CATALOG:
    raise RuntimeError("Duplicate catalog IDs.")

movie_index = {
    movie: i
    for i, movie in enumerate(catalog_ids)
}

N_CATALOG = len(catalog_ids)

print("Catalog items:", N_CATALOG)
print("Unique catalog IDs:", len(catalog_set))


# ============================================================
# 4. VALIDATE EVENT-CONVERSATION MAPPING
# ============================================================

print("\n========== 2. EVENT MAPPING VALIDATION ==========")

mapping_rows = read_csv(MAPPING_FILE)

event_to_conv = {}
mapping_targets = {}

for row in mapping_rows:

    eid = int(row["event_id"])
    cid = int(row["conv_id"])
    target = int(row["target_movie_id"])

    if eid in event_to_conv:
        raise RuntimeError(f"Duplicate event mapping: {eid}")

    if target not in catalog_set:
        raise RuntimeError(f"Invalid target movie: {target}")

    event_to_conv[eid] = cid
    mapping_targets[eid] = target


event_ids = sorted(event_to_conv)

conv_ids = sorted(set(event_to_conv.values()))

N_EVENTS = len(event_ids)
N_CONV = len(conv_ids)

if N_EVENTS != EXPECTED_EVENTS:
    raise RuntimeError("Unexpected event count.")

if N_CONV != EXPECTED_CONVERSATIONS:
    raise RuntimeError("Unexpected conversation count.")

if event_ids != list(range(1, EXPECTED_EVENTS + 1)):
    raise RuntimeError("Event IDs are not contiguous.")

conv_index = {
    cid: i
    for i, cid in enumerate(conv_ids)
}

event_conv = np.array(
    [conv_index[event_to_conv[eid]] for eid in event_ids],
    dtype=np.int32
)

print("Events:", N_EVENTS)
print("Conversations:", N_CONV)
print("Mapping integrity: PASS")


# ============================================================
# 5. LOAD AND VALIDATE EVENT DATA
# ============================================================

print("\n========== 3. EVENT DATA VALIDATION ==========")

rows = read_csv(EVENTS_FILE)

records = {}

METADATA_FIELDS = [
    "target_movie_id",
    "target_frequency",
    "frequency_group",
    "target_context_overlap",
    "context_nonempty"
]

event_metadata = {}

for row in rows:

    eid = int(row["event_id"])
    method = row["method"]

    if method not in METHODS:
        raise RuntimeError(f"Unexpected method: {method}")

    key = (eid, method)

    if key in records:
        raise RuntimeError(f"Duplicate event-method record: {key}")

    if eid not in event_to_conv:
        raise RuntimeError(f"Unmapped event: {eid}")

    target = int(row["target_movie_id"])

    if target != mapping_targets[eid]:
        raise RuntimeError(
            f"Target mismatch between mapping and event {eid}"
        )

    metadata = tuple(
        normalized_value(row[field])
        for field in METADATA_FIELDS
    )

    if eid in event_metadata:
        if event_metadata[eid] != metadata:
            raise RuntimeError(
                f"Method metadata mismatch for event {eid}"
            )
    else:
        event_metadata[eid] = metadata

    movies = list(
        map(int, json.loads(row["top10_movie_ids"]))
    )

    groups = list(
        map(int, json.loads(row["top10_group_ids"]))
    )

    if len(movies) != 10 or len(groups) != 10:
        raise RuntimeError(f"Invalid top-10 length: {key}")

    if len(set(movies)) != 10:
        raise RuntimeError(f"Duplicate recommendation: {key}")

    if not set(movies).issubset(catalog_set):
        raise RuntimeError(f"Out-of-catalog recommendation: {key}")

    if any(g not in range(5) for g in groups):
        raise RuntimeError(f"Invalid group: {key}")
    rank_value = row["rank_top50"]

    if rank_value.strip() != "":
        rank_value = int(float(rank_value))
    else:
        rank_value = None

    if rank_value is not None:
        if not 1 <= rank_value <= 50:
         raise RuntimeError(
            f"Invalid target rank for event {eid}, "
            f"method {method}: {rank_value}"
        )
    records[key] = {
    "movies": movies,
    "groups": groups,
    "rank_top50": rank_value,
    "hit10": float(row["hit10"]),
    "hit50": float(row["hit50"]),
    "ndcg10": float(row["ndcg10"]),
    "zero": int(float(row["target_frequency"]) == 0),
    "one": int(float(row["target_frequency"]) == 1),
    "overlap": int(
        float(row["target_context_overlap"]) == 1
    ),
    }


expected_keys = {
    (eid, method)
    for eid in event_ids
    for method in METHODS
}

if set(records) != expected_keys:
    missing = expected_keys - set(records)
    raise RuntimeError(
        f"Incomplete event-method records: {len(missing)} missing"
    )


# Validate frequency-group consistency for every recommended movie.
movie_groups = {}

for record in records.values():

    for movie, group in zip(
        record["movies"],
        record["groups"]
    ):

        if movie in movie_groups:
            if movie_groups[movie] != group:
                raise RuntimeError(
                    f"Movie {movie} has inconsistent frequency groups."
                )
        else:
            movie_groups[movie] = group


print("Event-method records:", len(records))
print("Expected records:", EXPECTED_EVENTS * len(METHODS))
print("Metadata consistency: PASS")
print("Movie-group consistency: PASS")


# ============================================================
# 6. PREPARE METHOD ARRAYS
# ============================================================

print("\n========== 4. ARRAY PREPARATION ==========")

arrays = {}

reference = [
    records[(eid, METHODS[0])]
    for eid in event_ids
]

zero_mask = np.array(
    [r["zero"] for r in reference],
    dtype=bool
)

one_mask = np.array(
    [r["one"] for r in reference],
    dtype=bool
)

overlap_mask = np.array(
    [r["overlap"] for r in reference],
    dtype=bool
)

nonoverlap_mask = ~overlap_mask

for method in METHODS:

    method_rows = [
        records[(eid, method)]
        for eid in event_ids
    ]

    arrays[method] = {

        "movie": np.array([
            [movie_index[x] for x in r["movies"]]
            for r in method_rows
        ], dtype=np.int32),

        "group": np.array([
            r["groups"] for r in method_rows
        ], dtype=np.int8),

        "hit10": np.array([
            r["hit10"] for r in method_rows
        ]),

        "hit50": np.array([
            r["hit50"] for r in method_rows
        ]),

        "ndcg10": np.array([
            r["ndcg10"] for r in method_rows
        ]),
    }

print("All six methods prepared.")


# ============================================================
# 7. METRIC CALCULATION
# ============================================================

def weighted_mean(values, weights, mask=None):

    if mask is not None:
        values = values[mask]
        weights = weights[mask]

    total = weights.sum()

    if total == 0:
        return np.nan

    return float(np.dot(values, weights) / total)


def calculate_metrics(method, weights):

    a = arrays[method]

    event_count = int(weights.sum())

    if event_count == 0:
        raise RuntimeError("Empty bootstrap sample.")

    movie_weights = np.repeat(weights, 10)

    flat_movies = a["movie"].ravel()

    exposure = np.bincount(
        flat_movies,
        weights=movie_weights,
        minlength=N_CATALOG
    )

    total_slots = exposure.sum()

    if not np.isclose(total_slots, event_count * 10):
        raise RuntimeError("Exposure slot mismatch.")

    coverage = np.count_nonzero(exposure) / N_CATALOG

    ordered = np.sort(exposure)

    coefficients = (
        2 * np.arange(1, N_CATALOG + 1)
        - N_CATALOG - 1
    )

    gini = (
        np.dot(coefficients, ordered)
        / (N_CATALOG * total_slots)
    )

    group_exposure = np.bincount(
        a["group"].ravel(),
        weights=movie_weights,
        minlength=5
    )

    group_shares = group_exposure / total_slots

    result = {

        "events": event_count,

        "hit10": weighted_mean(
            a["hit10"], weights
        ),

        "hit50": weighted_mean(
            a["hit50"], weights
        ),

        "ndcg10": weighted_mean(
            a["ndcg10"], weights
        ),

        "coverage10": float(coverage),

        "item_gini": float(gini),

        "zero_hit10": weighted_mean(
            a["hit10"], weights, zero_mask
        ),

        "one_hit10": weighted_mean(
            a["hit10"], weights, one_mask
        ),

        "overlap_hit10": weighted_mean(
            a["hit10"], weights, overlap_mask
        ),

        "nonoverlap_hit10": weighted_mean(
            a["hit10"], weights, nonoverlap_mask
        ),
    }

    for i in range(5):
        result[f"group_{i}_share"] = float(
            group_shares[i]
        )

    return result


METRICS = [
    "hit10",
    "hit50",
    "ndcg10",
    "coverage10",
    "item_gini",
    "zero_hit10",
    "one_hit10",
    "overlap_hit10",
    "nonoverlap_hit10",
] + [
    f"group_{i}_share"
    for i in range(5)
]


# ============================================================
# 8. REPRODUCE FULL-TEST RESULTS
# ============================================================

print("\n========== 5. FULL-TEST REPRODUCTION ==========")

full_weights = np.ones(
    N_EVENTS,
    dtype=np.int64
)

full_results = {
    method: calculate_metrics(method, full_weights)
    for method in METHODS
}

for method in METHODS:

    result = full_results[method]

    print(
        f"{method:25s}"
        f" Hit@10={result['hit10']:.6f}"
        f" NDCG@10={result['ndcg10']:.6f}"
        f" Coverage={result['coverage10']:.6f}"
        f" Gini={result['item_gini']:.6f}"
    )


# ============================================================
# 9. LOAD PREVIOUS EXPERIMENT 11 RESULTS
# ============================================================

print("\n========== 6. EXPERIMENT 11 REPRODUCTION CHECK ==========")

previous_rows = read_csv(PREVIOUS_RESULTS)

previous = {
    (r["method"], r["metric"]): r
    for r in previous_rows
}

if len(previous) != len(METHODS) * len(METRICS):
    raise RuntimeError("Unexpected Experiment 11 result structure.")

for method in METHODS:

    for metric in METRICS:

        saved = previous[(method, metric)]

        calculated = full_results[method][metric]

        if not np.isclose(
            calculated,
            float(saved["full_test"]),
            atol=1e-10,
            rtol=0
        ):
            raise RuntimeError(
                f"Full-test mismatch: {method}/{metric}"
            )

print("Full-test metrics: PASS")
print("Experiment 11 reference: loaded")


# ============================================================
# 10. CONVERSATION-CLUSTERED BOOTSTRAP
# ============================================================

print("\n========== 7. BOOTSTRAP REPRODUCTION ==========")

bootstrap = {
    method: {
        metric: np.full(B, np.nan)
        for metric in METRICS
    }
    for method in METHODS
}

for b in range(B):

    sampled = rng.integers(
        0,
        N_CONV,
        size=N_CONV
    )

    cluster_multiplicity = np.bincount(
        sampled,
        minlength=N_CONV
    )

    event_weights = cluster_multiplicity[event_conv]

    for method in METHODS:

        metrics = calculate_metrics(
            method,
            event_weights
        )

        for metric in METRICS:
            bootstrap[method][metric][b] = metrics[metric]

    if (b + 1) % 250 == 0:
        print(f"Completed {b + 1}/{B}")


# ============================================================
# 11. DISTRIBUTION AND BIAS DIAGNOSTICS
# ============================================================

print("\n========== 8. DISTRIBUTION DIAGNOSTICS ==========")

distribution_rows = []
interval_rows = []

for method in METHODS:

    for metric in METRICS:

        theta = full_results[method][metric]

        values = bootstrap[method][metric]

        valid = values[np.isfinite(values)]

        if len(valid) < B * 0.99:
            raise RuntimeError(
                f"Insufficient valid replicates: {method}/{metric}"
            )

        mean = float(np.mean(valid))
        bias = mean - theta
        sd = float(np.std(valid, ddof=1))

        q_low, q_high = percentile_interval(valid)

        # Basic bootstrap interval:
        # [2*theta - q97.5, 2*theta - q2.5]
        basic_low = 2 * theta - q_high
        basic_high = 2 * theta - q_low

        bias_ratio = (
            bias / sd if sd > 0 else np.nan
        )

        distribution_rows.append({

            "method": method,
            "metric": metric,
            "full_test": theta,
            "bootstrap_mean": mean,
            "bootstrap_bias": bias,
            "bootstrap_sd": sd,
            "bias_to_sd_ratio": bias_ratio,
            "bootstrap_median": float(np.median(valid)),
            "bootstrap_min": float(np.min(valid)),
            "bootstrap_max": float(np.max(valid)),
            "valid_replicates": len(valid),
        })

        interval_rows.append({

            "method": method,
            "metric": metric,
            "full_test": theta,

            "percentile_lower": float(q_low),
            "percentile_upper": float(q_high),

            "basic_lower": float(basic_low),
            "basic_upper": float(basic_high),

            "percentile_contains_full_test": bool(
                q_low <= theta <= q_high
            ),

            "basic_contains_full_test": bool(
                basic_low <= theta <= basic_high
            ),

            "bootstrap_bias": bias,
            "bootstrap_sd": sd,
        })


save_csv(
    OUT / "bootstrap_distribution_diagnostics.csv",
    distribution_rows
)

save_csv(
    OUT / "bootstrap_interval_diagnostics.csv",
    interval_rows
)


# ============================================================
# 12. VALIDATE EXPERIMENT 11 BOOTSTRAP RESULTS
# ============================================================

print("\n========== 9. EXPERIMENT 11 BOOTSTRAP VALIDATION ==========")

for method in METHODS:

    for metric in METRICS:

        saved = previous[(method, metric)]

        valid = bootstrap[method][metric]
        valid = valid[np.isfinite(valid)]

        checks = {

            "bootstrap_mean": float(np.mean(valid)),

            "ci_lower": float(np.percentile(valid, 2.5)),

            "ci_upper": float(np.percentile(valid, 97.5)),
        }

        for key, value in checks.items():

            if not np.isclose(
                value,
                float(saved[key]),
                atol=1e-9,
                rtol=0
            ):
                raise RuntimeError(
                    f"Experiment 11 mismatch: "
                    f"{method}/{metric}/{key}"
                )

print("All previous bootstrap summaries reproduced: PASS")


# ============================================================
# 13. PAIRED COMPARISON VALIDATION
# ============================================================

print("\n========== 10. PAIRED COMPARISONS ==========")

comparison_rows = []

for proposed, baseline in COMPARISONS:

    for metric in METRICS:

        delta_hat = (
            full_results[proposed][metric]
            - full_results[baseline][metric]
        )

        delta_boot = (
            bootstrap[proposed][metric]
            - bootstrap[baseline][metric]
        )

        valid = delta_boot[np.isfinite(delta_boot)]

        if len(valid) < B * 0.99:
            raise RuntimeError(
                f"Insufficient paired replicates: {metric}"
            )

        ci_low, ci_high = percentile_interval(valid)

        # Centered-bootstrap approximate two-sided p-value.
        centered = valid - delta_hat

        p_value = (
            1 + np.sum(
                np.abs(centered) >= abs(delta_hat)
            )
        ) / (len(valid) + 1)

        comparison_rows.append({

            "proposed": proposed,
            "baseline": baseline,
            "metric": metric,

            "full_test_difference": float(delta_hat),

            "bootstrap_ci_lower": float(ci_low),
            "bootstrap_ci_upper": float(ci_high),

            "bootstrap_p_raw": float(p_value),

            "positive_fraction": float(
                np.mean(valid > 0)
            ),

            "negative_fraction": float(
                np.mean(valid < 0)
            ),

            "valid_replicates": len(valid),
        })


# Holm adjustment within each metric family.
for metric in METRICS:

    selected = [
        row for row in comparison_rows
        if row["metric"] == metric
    ]

    adjusted = holm_adjust([
        row["bootstrap_p_raw"]
        for row in selected
    ])

    for row, adjusted_p in zip(selected, adjusted):
        row["bootstrap_p_holm"] = float(adjusted_p)


save_csv(
    OUT / "bootstrap_paired_comparisons.csv",
    comparison_rows
)


# ============================================================
# 14. COVERAGE AND GINI INVESTIGATION
# ============================================================

print("\n========== 11. COVERAGE AND GINI INVESTIGATION ==========")

special_rows = []

for row in distribution_rows:

    if row["metric"] not in ("coverage10", "item_gini"):
        continue

    method = row["method"]

    theta = row["full_test"]
    mean = row["bootstrap_mean"]
    bias = row["bootstrap_bias"]
    sd = row["bootstrap_sd"]

    special_rows.append({

        "method": method,
        "metric": row["metric"],
        "full_test": theta,
        "bootstrap_mean": mean,
        "bias": bias,
        "sd": sd,
        "bias_to_sd_ratio": row["bias_to_sd_ratio"],
    })

    print(
        f"{method:25s}"
        f" {row['metric']:12s}"
        f" Full={theta:.6f}"
        f" Mean={mean:.6f}"
        f" Bias={bias:+.6f}"
        f" SD={sd:.6f}"
    )


# ============================================================
# 15. SAVE RAW BOOTSTRAP REPLICATES
# ============================================================

print("\n========== 12. SAVING BOOTSTRAP DISTRIBUTIONS ==========")

np.savez_compressed(

    OUT / "bootstrap_replicates.npz",

    **{
        f"{method}__{metric}": bootstrap[method][metric]
        for method in METHODS
        for metric in METRICS
    }
)

print("Raw bootstrap distributions saved.")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

summary = {

    "experiment": "bootstrap_inference_validation",

    "reference_experiment": "conversation_clustered_bootstrap",

    "split": "held_out_test",

    "sampling_unit": "conversation",

    "sampling": "with_replacement",

    "replicates": B,

    "seed": SEED,

    "catalog_size": N_CATALOG,

    "events": N_EVENTS,

    "conversations": N_CONV,

    "methods": METHODS,

    "comparisons": [
        {
            "proposed": p,
            "baseline": b
        }
        for p, b in COMPARISONS
    ],

    "metric_count": len(METRICS),

    "full_test_reproduction": "PASS",

    "experiment_11_bootstrap_reproduction": "PASS",

    "coverage_and_gini_diagnostics": special_rows,

    "interval_methods": [
        "percentile_bootstrap",
        "basic_bootstrap"
    ],

    "p_value_method": (
        "Centered paired-bootstrap approximation"
    ),

    "multiple_testing": (
        "Holm correction within each metric family"
    ),

    "monte_carlo_p_value_resolution": 1.0 / (B + 1),

    "interpretation": (
        "Bootstrap bias and interval diagnostics are descriptive. "
        "The percentile and basic intervals are compared, but neither "
        "is automatically assumed to provide calibrated coverage. "
        "Further assessment is required before treating these "
        "intervals or approximate p-values as definitive inferential evidence."
    ),

    "outputs": OUTPUT_FILES,
}

with (
    OUT / "bootstrap_validation_summary.json"
).open("w", encoding="utf-8") as f:

    json.dump(summary, f, indent=2)


# ============================================================
# 17. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT 12 COMPLETED")
print("=" * 65)

print("Catalog:", N_CATALOG)
print("Events:", N_EVENTS)
print("Conversations:", N_CONV)
print("Methods:", len(METHODS))
print("Bootstrap replicates:", B)
print("Metric families:", len(METRICS))
print("Paired comparisons:", len(COMPARISONS))

print("\nValidation:")
print("Full-test reproduction: PASS")
print("Experiment 11 reproduction: PASS")
print("Metadata integrity: PASS")
print("Movie-group consistency: PASS")

print("\nOutput directory:", OUT.resolve())

for filename in OUTPUT_FILES:
    path = OUT / filename
    print(f"{filename}: {path.stat().st_size:,} bytes")

print("\nNo model training or inference performed.")
print("EXPERIMENT 12 COMPLETE.")
