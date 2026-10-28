"""Held-out item-exposure sensitivity analysis for KG-FairCRS.

Uses saved top-10 lists, without model inference.
Subsamples 80% of conversations WITHOUT replacement.
Subsample percentiles are descriptive, NOT confidence intervals.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("results_final_test_m2")
EVENTS = ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"
MAPPING = ROOT / "m2_test_event_conversation_mapping.csv"
REPORT = ROOT / "adaptive_m2_with_kgsf_baseline.json"

OUTPUT = ROOT / "m2_test_exposure_sensitivity.csv"
SUMMARY = ROOT / "m2_test_exposure_sensitivity_summary.json"

METHODS = (
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_3",
)
BASELINE = "KGSF_only"
PROPOSED = "A1_frequency_3"

REPS = 500
FRACTION = 0.80
SEED = 20261029

for path in (OUTPUT, SUMMARY):
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}"
        )

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

with REPORT.open(encoding="utf-8") as f:
    report = json.load(f)

catalog_size = int(report["candidate_movies"])

if report["events"] != 4003 or catalog_size != 6924:
    raise RuntimeError("Unexpected frozen test metadata.")

reference = {
    item["method"]: item
    for item in report["results"]
}

map_rows = read_csv(MAPPING)
mapping = {}

for r in map_rows:
    eid = int(r["event_id"])
    if eid in mapping:
        raise RuntimeError(f"Duplicate mapping event: {eid}")
    mapping[eid] = r["conv_id"]

if set(mapping) != set(range(1, 4004)):
    raise RuntimeError("Incomplete event-conversation mapping.")

rows = read_csv(EVENTS)
records = {}

for r in rows:
    method = r["method"]
    if method not in METHODS:
        continue

    eid = int(r["event_id"])
    key = (eid, method)

    if key in records:
        raise RuntimeError(f"Duplicate event-method: {key}")

    movies = json.loads(r["top10_movie_ids"])
    groups = json.loads(r["top10_group_ids"])

    if len(movies) != 10 or len(groups) != 10:
        raise RuntimeError(f"Invalid top-10 list: {key}")

    if len(set(movies)) != 10:
        raise RuntimeError(f"Duplicate movie in top-10: {key}")

    if any(int(g) not in range(5) for g in groups):
        raise RuntimeError(f"Invalid popularity group: {key}")

    records[key] = (
        tuple(int(x) for x in movies),
        tuple(int(x) for x in groups),
    )

expected_keys = {
    (eid, method)
    for eid in mapping
    for method in METHODS
}

if set(records) != expected_keys:
    raise RuntimeError("Incomplete method-event records.")

# Check the same movie always carries the same popularity group.
movie_group = {}

for movies, groups in records.values():
    for movie, group in zip(movies, groups):
        if movie in movie_group and movie_group[movie] != group:
            raise RuntimeError(
                f"Inconsistent popularity group for movie {movie}"
            )
        movie_group[movie] = group

if len(movie_group) > catalog_size:
    raise RuntimeError("Exposed items exceed catalog size.")

movie_index = {
    movie: i
    for i, movie in enumerate(sorted(movie_group))
}

event_ids = sorted(mapping)
conv_ids = sorted(set(mapping.values()))

conv_position = {
    cid: i for i, cid in enumerate(conv_ids)
}

event_conv = np.array(
    [conv_position[mapping[eid]] for eid in event_ids],
    dtype=np.int32
)

movie_arrays = {}
group_arrays = {}

for method in METHODS:
    movie_arrays[method] = np.array(
        [
            [
                movie_index[x]
                for x in records[(eid, method)][0]
            ]
            for eid in event_ids
        ],
        dtype=np.int32,
    )

    group_arrays[method] = np.array(
        [
            records[(eid, method)][1]
            for eid in event_ids
        ],
        dtype=np.int32,
    )

def exposure_metrics(method, selected):
    movie_lists = movie_arrays[method][selected]
    group_lists = group_arrays[method][selected]

    if len(movie_lists) == 0:
        raise RuntimeError("Empty conversation subsample.")

    counts = np.bincount(
        movie_lists.ravel(),
        minlength=catalog_size
    ).astype(np.float64)

    total = float(counts.sum())
    expected = len(movie_lists) * 10

    if total != expected:
        raise RuntimeError("Exposure slot mismatch.")

    coverage = float(np.count_nonzero(counts) / catalog_size)

    # Standard Gini over every eligible catalog item,
    # including items with zero exposure.
    ordered = np.sort(counts)
    n = len(ordered)

    coefficients = 2 * np.arange(1, n + 1) - n - 1
    gini = float(
        np.dot(coefficients, ordered) / (n * total)
    )

    group_counts = np.bincount(
        group_lists.ravel(),
        minlength=5
    )
    shares = group_counts / total

    return {
        "coverage10": coverage,
        "item_gini": gini,
        "group_shares": shares.tolist(),
        "events": int(len(movie_lists)),
    }

all_selected = np.ones(len(event_ids), dtype=bool)
full = {
    method: exposure_metrics(method, all_selected)
    for method in METHODS
}

print("=== VERIFY FULL HELD-OUT EXPOSURE METRICS ===")

for method in METHODS:
    computed = full[method]
    saved = reference[method]

    for metric, report_key in (
        ("coverage10", "catalog_coverage10"),
        ("item_gini", "item_gini"),
    ):
        if not np.isclose(
            computed[metric],
            saved[report_key],
            rtol=0,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"{method}/{metric}: "
                f"recomputed={computed[metric]}, "
                f"saved={saved[report_key]}"
            )

    if not np.allclose(
        computed["group_shares"],
        saved["group_exposure_shares"],
        atol=1e-9,
        rtol=0,
    ):
        raise RuntimeError(
            f"{method}: group exposure shares differ."
        )

    print(
        f"{method:18s} "
        f"coverage={computed['coverage10']:.6f} "
        f"gini={computed['item_gini']:.6f} PASS"
    )

rng = np.random.default_rng(SEED)
sample_size = round(FRACTION * len(conv_ids))

results = []

print("\n=== WITHOUT-REPLACEMENT SENSITIVITY ===")
print("Conversations:", len(conv_ids))
print("Selected per replicate:", sample_size)
print("Replicates:", REPS)
print("Seed:", SEED)

for rep in range(REPS):
    sampled_conv = rng.choice(
        len(conv_ids),
        size=sample_size,
        replace=False,
    )

    selected = np.isin(event_conv, sampled_conv)

    for method in METHODS:
        metrics = exposure_metrics(method, selected)

        result = {
            "replicate": rep + 1,
            "method": method,
            "sample_fraction": FRACTION,
            "n_conversations": sample_size,
            "n_events": metrics["events"],
            "coverage10": metrics["coverage10"],
            "item_gini": metrics["item_gini"],
        }

        for i, share in enumerate(metrics["group_shares"]):
            result[f"group_{i}_share"] = share

        results.append(result)

# Summarize matched subsample differences.
indexed = {
    (r["replicate"], r["method"]): r
    for r in results
}

comparisons = []

for baseline in ("KGSF_only", "A0_fixed_025"):
    for metric in ("coverage10", "item_gini"):
        deltas = np.array([
            indexed[(rep, PROPOSED)][metric]
            - indexed[(rep, baseline)][metric]
            for rep in range(1, REPS + 1)
        ])

        full_delta = (
            full[PROPOSED][metric]
            - full[baseline][metric]
        )

        comparisons.append({
            "proposed": PROPOSED,
            "baseline": baseline,
            "metric": metric,
            "full_test_delta": float(full_delta),
            "subsample_mean_delta": float(deltas.mean()),
            "subsample_p05": float(np.percentile(deltas, 5)),
            "subsample_p50": float(np.percentile(deltas, 50)),
            "subsample_p95": float(np.percentile(deltas, 95)),
            "fraction_positive": float(np.mean(deltas > 0)),
            "fraction_negative": float(np.mean(deltas < 0)),
        })

        print(
            f"\n{PROPOSED} vs {baseline}, {metric}"
        )
        print(f"  Full-test difference: {full_delta:+.6f}")
        print(
            "  Subsample P05/P50/P95:",
            f"{np.percentile(deltas, 5):+.6f}",
            f"{np.percentile(deltas, 50):+.6f}",
            f"{np.percentile(deltas, 95):+.6f}",
        )
        print(
            "  Positive fraction:",
            f"{np.mean(deltas > 0):.3f}"
        )

with OUTPUT.open(
    "x", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0])
    )
    writer.writeheader()
    writer.writerows(results)

summary = {
    "split": "held_out_test",
    "analysis": "without_replacement_conversation_subsampling",
    "interpretation": (
        "Subsample percentiles are descriptive sensitivity "
        "ranges, not confidence intervals or p-values."
    ),
    "sample_fraction": FRACTION,
    "replicates": REPS,
    "seed": SEED,
    "catalog_size": catalog_size,
    "full_test": full,
    "comparisons": comparisons,
}

with SUMMARY.open("x", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nPASS: Full-test metrics reproduced exactly.")
print("Saved:", OUTPUT)
print("Saved:", SUMMARY)
print("No model inference or training performed.")
