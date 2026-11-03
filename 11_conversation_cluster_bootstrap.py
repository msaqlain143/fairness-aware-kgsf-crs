
import csv
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

# ============================================================
# EXPERIMENT 11: CONVERSATION-CLUSTERED BOOTSTRAP
# ============================================================

ROOT = Path("results_final_test_m2")
CATALOG_FILE = Path("catalog_audit_corrected.pkl")

EVENTS_FILE = ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"
MAPPING_FILE = ROOT / "m2_test_event_conversation_mapping.csv"

OUT = Path("results_cluster_bootstrap")
OUT.mkdir(exist_ok=True)

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

rng = np.random.default_rng(SEED)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def holm_adjust(pvalues):
    """Holm step-down correction."""
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


# ------------------------------------------------------------
# 1. LOAD AND VALIDATE CATALOG
# ------------------------------------------------------------

with open(CATALOG_FILE, "rb") as f:
    catalog = pickle.load(f)

catalog_ids = list(map(int, catalog["movie_ids"]))
catalog_set = set(catalog_ids)

if len(catalog_ids) != 6924 or len(catalog_set) != 6924:
    raise RuntimeError("Unexpected catalog size or duplicate IDs.")

movie_index = {
    movie: i for i, movie in enumerate(catalog_ids)
}

N_CATALOG = len(catalog_ids)


# ------------------------------------------------------------
# 2. LOAD EVENT-CONVERSATION MAPPING
# ------------------------------------------------------------

mapping_rows = read_csv(MAPPING_FILE)

event_to_conv = {}

for row in mapping_rows:
    eid = int(row["event_id"])
    cid = int(row["conv_id"])

    if eid in event_to_conv:
        raise RuntimeError(f"Duplicate event mapping: {eid}")

    event_to_conv[eid] = cid

event_ids = sorted(event_to_conv)
conv_ids = sorted(set(event_to_conv.values()))

conv_index = {
    cid: i for i, cid in enumerate(conv_ids)
}

event_conv = np.array(
    [conv_index[event_to_conv[eid]] for eid in event_ids],
    dtype=np.int32
)

N_EVENTS = len(event_ids)
N_CONV = len(conv_ids)


# ------------------------------------------------------------
# 3. LOAD EVENT DATA
# ------------------------------------------------------------

rows = read_csv(EVENTS_FILE)

records = {}

for row in rows:
    eid = int(row["event_id"])
    method = row["method"]

    key = (eid, method)

    if key in records:
        raise RuntimeError(f"Duplicate record: {key}")

    if method not in METHODS:
        continue

    movies = json.loads(row["top10_movie_ids"])
    groups = json.loads(row["top10_group_ids"])

    if len(movies) != 10 or len(groups) != 10:
        raise RuntimeError(f"Invalid top-10: {key}")

    if len(set(movies)) != 10:
        raise RuntimeError(f"Duplicate top-10 movie: {key}")

    if not set(movies).issubset(catalog_set):
        raise RuntimeError(f"Recommendation outside catalog: {key}")

    if any(int(g) not in range(5) for g in groups):
        raise RuntimeError(f"Invalid group: {key}")

    records[key] = {
        "movies": movies,
        "groups": groups,
        "hit10": float(row["hit10"]),
        "hit50": float(row["hit50"]),
        "ndcg10": float(row["ndcg10"]),
        "zero": int(float(row["target_frequency"]) == 0),
        "one": int(float(row["target_frequency"]) == 1),
        "overlap": int(float(row["target_context_overlap"]) == 1),
    }

expected = {
    (eid, method)
    for eid in event_ids
    for method in METHODS
}

if set(records) != expected:
    raise RuntimeError("Incomplete event-method records.")

print("Data integrity verified.")
print("Catalog:", N_CATALOG)
print("Events:", N_EVENTS)
print("Conversations:", N_CONV)
print("Methods:", len(METHODS))


# ------------------------------------------------------------
# 4. PREPARE METHOD ARRAYS
# ------------------------------------------------------------

arrays = {}

for method in METHODS:

    method_rows = [
        records[(eid, method)]
        for eid in event_ids
    ]

    movie = np.array([
        [movie_index[x] for x in r["movies"]]
        for r in method_rows
    ], dtype=np.int32)

    group = np.array([
        r["groups"] for r in method_rows
    ], dtype=np.int8)

    arrays[method] = {
        "movie": movie,
        "group": group,
        "hit10": np.array([r["hit10"] for r in method_rows]),
        "hit50": np.array([r["hit50"] for r in method_rows]),
        "ndcg10": np.array([r["ndcg10"] for r in method_rows]),
    }


# ------------------------------------------------------------
# 5. EVENT SUBGROUP MASKS
# ------------------------------------------------------------

reference = [records[(eid, METHODS[0])] for eid in event_ids]

zero_mask = np.array([r["zero"] for r in reference], dtype=bool)
one_mask = np.array([r["one"] for r in reference], dtype=bool)
overlap_mask = np.array([r["overlap"] for r in reference], dtype=bool)

nonoverlap_mask = ~overlap_mask


# ------------------------------------------------------------
# 6. METRIC CALCULATION
# ------------------------------------------------------------

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

    # Full-catalog exposure counts, including zero-exposure items.
    flat_movies = a["movie"].ravel()

    movie_weights = np.repeat(weights, 10)

    exposure = np.bincount(
        flat_movies,
        weights=movie_weights,
        minlength=N_CATALOG
    )

    total_slots = exposure.sum()

    coverage = np.count_nonzero(exposure) / N_CATALOG

    ordered = np.sort(exposure)
    n = N_CATALOG

    coefficients = 2 * np.arange(1, n + 1) - n - 1

    gini = (
        np.dot(coefficients, ordered)
        / (n * total_slots)
    )

    # Group exposure shares.
    group_exposure = np.zeros(5)

    np.add.at(
        group_exposure,
        a["group"].ravel(),
        movie_weights
    )

    group_shares = group_exposure / total_slots

    result = {
        "events": event_count,
        "hit10": weighted_mean(a["hit10"], weights),
        "hit50": weighted_mean(a["hit50"], weights),
        "ndcg10": weighted_mean(a["ndcg10"], weights),
        "coverage10": float(coverage),
        "item_gini": float(gini),
    }

    result["zero_hit10"] = weighted_mean(
        a["hit10"], weights, zero_mask
    )

    result["one_hit10"] = weighted_mean(
        a["hit10"], weights, one_mask
    )

    result["overlap_hit10"] = weighted_mean(
        a["hit10"], weights, overlap_mask
    )

    result["nonoverlap_hit10"] = weighted_mean(
        a["hit10"], weights, nonoverlap_mask
    )

    for i in range(5):
        result[f"group_{i}_share"] = float(group_shares[i])

    return result


# ------------------------------------------------------------
# 7. FULL TEST METRICS
# ------------------------------------------------------------

full_weights = np.ones(N_EVENTS, dtype=np.int64)

full_results = {
    method: calculate_metrics(method, full_weights)
    for method in METHODS
}

print("\n========== FULL TEST RESULTS ==========")

for method, metrics in full_results.items():
    print(
        method,
        "| Hit@10:", round(metrics["hit10"], 5),
        "| NDCG@10:", round(metrics["ndcg10"], 5),
        "| Coverage:", round(metrics["coverage10"], 5),
        "| Gini:", round(metrics["item_gini"], 5),
    )


# ------------------------------------------------------------
# 8. CONVERSATION-CLUSTERED BOOTSTRAP
# ------------------------------------------------------------

metric_names = [
    "hit10",
    "hit50",
    "ndcg10",
    "coverage10",
    "item_gini",
    "zero_hit10",
    "one_hit10",
    "overlap_hit10",
    "nonoverlap_hit10",
] + [f"group_{i}_share" for i in range(5)]

bootstrap = {
    method: {
        metric: np.full(B, np.nan)
        for metric in metric_names
    }
    for method in METHODS
}

print("\n========== BOOTSTRAP ==========")
print("Replicates:", B)
print("Conversation clusters:", N_CONV)
print("Sampling: with replacement")

for b in range(B):

    # Sample conversations, not individual events.
    sampled = rng.integers(0, N_CONV, size=N_CONV)

    cluster_multiplicity = np.bincount(
        sampled,
        minlength=N_CONV
    )

    event_weights = cluster_multiplicity[event_conv]

    for method in METHODS:

        metrics = calculate_metrics(method, event_weights)

        for metric in metric_names:
            bootstrap[method][metric][b] = metrics[metric]

    if (b + 1) % 250 == 0:
        print(f"Completed {b + 1}/{B}")


# ------------------------------------------------------------
# 9. PAIRED BOOTSTRAP COMPARISONS
# ------------------------------------------------------------

comparison_rows = []

for proposed, baseline in COMPARISONS:

    for metric in metric_names:

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
                f"Too many undefined replicates: {metric}"
            )

        ci_low, ci_high = np.percentile(
            valid, [2.5, 97.5]
        )

        # Centered paired-bootstrap approximation.
        centered = valid - delta_hat

        p_value = (
            1 + np.sum(np.abs(centered) >= abs(delta_hat))
        ) / (len(valid) + 1)

        comparison_rows.append({
            "proposed": proposed,
            "baseline": baseline,
            "metric": metric,
            "full_test_difference": float(delta_hat),
            "bootstrap_ci_lower": float(ci_low),
            "bootstrap_ci_upper": float(ci_high),
            "bootstrap_p_raw": float(p_value),
            "valid_replicates": len(valid),
        })


# ------------------------------------------------------------
# 10. HOLM CORRECTION WITHIN METRIC FAMILIES
# ------------------------------------------------------------

for metric in metric_names:

    selected = [
        r for r in comparison_rows
        if r["metric"] == metric
    ]

    adjusted = holm_adjust([
        r["bootstrap_p_raw"] for r in selected
    ])

    for row, p_adj in zip(selected, adjusted):
        row["bootstrap_p_holm"] = float(p_adj)


# ------------------------------------------------------------
# 11. SAVE OUTPUTS
# ------------------------------------------------------------

def save_csv(path, rows):

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


method_rows = []

for method in METHODS:

    for metric in metric_names:

        values = bootstrap[method][metric]
        valid = values[np.isfinite(values)]

        method_rows.append({
            "method": method,
            "metric": metric,
            "full_test": full_results[method][metric],
            "bootstrap_mean": float(np.mean(valid)),
            "ci_lower": float(np.percentile(valid, 2.5)),
            "ci_upper": float(np.percentile(valid, 97.5)),
        })


save_csv(
    OUT / "cluster_bootstrap_results.csv",
    method_rows
)

save_csv(
    OUT / "cluster_bootstrap_comparisons.csv",
    comparison_rows
)

summary = {
    "experiment": "conversation_clustered_bootstrap",
    "split": "held_out_test",
    "sampling_unit": "conversation",
    "sampling": "with_replacement",
    "replicates": B,
    "seed": SEED,
    "catalog_size": N_CATALOG,
    "events": N_EVENTS,
    "conversations": N_CONV,
    "methods": METHODS,
    "comparisons": comparison_rows,
    "interpretation": (
        "Paired conversation-cluster bootstrap. "
        "Percentile intervals describe bootstrap uncertainty. "
        "P-values are approximate centered-bootstrap values; "
        "Holm correction is applied within each metric family."
    ),
}

with (OUT / "cluster_bootstrap_summary.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n========== EXPERIMENT 11 COMPLETE ==========")
print("Results:", OUT.resolve())
print("Replicates:", B)
print("Methods:", len(METHODS))
print("Comparisons:", len(COMPARISONS))
print("Metric families:", len(metric_names))
print("No model inference performed.")
