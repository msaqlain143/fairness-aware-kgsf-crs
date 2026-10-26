from pathlib import Path
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results_corrected" / "adaptive_m2_validation_events_clustered.csv"
OUT = ROOT / "results_corrected"

BASELINE = "A0_fixed_025"
METRICS = ("hit10", "hit50", "ndcg10")
REPS = 10000
SEED = 2026


def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input CSV is empty.")

    required = {
        "conv_id", "event_id", "method",
        "hit10", "hit50", "ndcg10",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    methods = sorted({r["method"] for r in rows})
    if BASELINE not in methods:
        raise ValueError(f"Baseline not found: {BASELINE}")

    # One event must belong to one conversation and have one row per method.
    events = {}
    for r in rows:
        key = r["event_id"]
        conv = r["conv_id"]
        method = r["method"]

        if not conv or not key:
            raise ValueError("Missing conversation or event ID.")

        if key not in events:
            events[key] = {"conv_id": conv, "methods": {}}

        e = events[key]
        if e["conv_id"] != conv:
            raise ValueError(f"Event {key} maps to multiple conversations.")

        if method in e["methods"]:
            raise ValueError(f"Duplicate method {method} for event {key}.")

        e["methods"][method] = r

    expected_methods = set(methods)
    for key, e in events.items():
        if set(e["methods"]) != expected_methods:
            raise ValueError(f"Incomplete method set for event {key}.")

    conversations = {}
    for event_id, e in events.items():
        conv = e["conv_id"]
        conversations.setdefault(conv, [])
        conversations[conv].append(e)

    conv_ids = sorted(conversations)
    n_events = len(events)

    print("Conversations:", len(conv_ids))
    print("Events:", n_events)
    print("Methods:", len(methods))
    print("Rows:", len(rows))

    if len(rows) != n_events * len(methods):
        raise ValueError("Row count does not equal events times methods.")

    # Sum metric values within each conversation and method.
    # Sampling these sums preserves the event clustering.
    sums = np.zeros(
        (len(conv_ids), len(methods), len(METRICS)),
        dtype=np.float64,
    )
    counts = np.zeros(len(conv_ids), dtype=np.int64)

    method_pos = {m: i for i, m in enumerate(methods)}

    for ci, conv in enumerate(conv_ids):
        for e in conversations[conv]:
            counts[ci] += 1
            for method in methods:
                r = e["methods"][method]
                mi = method_pos[method]
                for ki, metric in enumerate(METRICS):
                    value = float(r[metric])
                    if not np.isfinite(value):
                        raise ValueError(
                            f"Non-finite {metric} in event {e}"
                        )
                    sums[ci, mi, ki] += value

    total_events = int(counts.sum())
    if total_events != n_events:
        raise ValueError("Conversation event counts do not reconcile.")

    observed = sums.sum(axis=0) / total_events
    baseline_idx = method_pos[BASELINE]
    candidates = [m for m in methods if m != BASELINE]

    rng = np.random.default_rng(SEED)
    differences = {
        (method, metric): np.empty(REPS, dtype=np.float64)
        for method in candidates
        for metric in METRICS
    }

    # Draw the same conversation sample for every method in each replicate.
    for b in range(REPS):
        sampled = rng.integers(0, len(conv_ids), size=len(conv_ids))
        denominator = int(counts[sampled].sum())
        replicate_means = sums[sampled].sum(axis=0) / denominator

        for method in candidates:
            mi = method_pos[method]
            for ki, metric in enumerate(METRICS):
                differences[(method, metric)][b] = (
                    replicate_means[mi, ki]
                    - replicate_means[baseline_idx, ki]
                )

    results = []
    for method in candidates:
        mi = method_pos[method]
        for ki, metric in enumerate(METRICS):
            delta = observed[mi, ki] - observed[baseline_idx, ki]
            draws = differences[(method, metric)]
            low, high = np.quantile(draws, [0.025, 0.975])

            results.append({
                "baseline": BASELINE,
                "method": method,
                "metric": metric,
                "baseline_mean": float(observed[baseline_idx, ki]),
                "candidate_mean": float(observed[mi, ki]),
                "difference_candidate_minus_baseline": float(delta),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_replicates": REPS,
                "seed": SEED,
                "unit_resampled": "conversation",
                "ci_excludes_zero": bool(low > 0 or high < 0),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "m2_cluster_bootstrap_summary.csv"
    json_path = OUT / "m2_cluster_bootstrap_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    report = {
        "split": "validation_only",
        "unit_resampled": "conversation",
        "conversations": len(conv_ids),
        "events": n_events,
        "methods": methods,
        "baseline": BASELINE,
        "metrics": list(METRICS),
        "bootstrap_replicates": REPS,
        "seed": SEED,
        "results": results,
        "interpretation_note": (
            "Percentile confidence intervals for paired candidate-minus-baseline "
            "differences. These are not multiple-comparison-adjusted. "
            "A confidence interval excluding zero is evidence of a difference "
            "under this bootstrap procedure, not proof of scientific novelty."
        ),
    }

    json_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print("\nCandidate-minus-baseline results:")
    for r in results:
        print(
            f"{r['method']:25s} {r['metric']:6s} "
            f"delta={r['difference_candidate_minus_baseline']:+.6f} "
            f"95% CI=[{r['ci95_low']:+.6f}, "
            f"{r['ci95_high']:+.6f}]"
        )


if __name__ == "__main__":
    main()
