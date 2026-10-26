"""
Paired bootstrap analysis for adaptive M2 validation results.

Input:
  results_corrected/adaptive_m2_validation_events.csv

The script compares each method with A0_fixed_025 using paired event-level
bootstrap resampling. It does not access the test split or modify model files.

Important:
  This CSV contains per-event ranking metrics, but not the identities of the
  recommended top-10 movies. Therefore this script estimates uncertainty for
  Hit@10, Hit@50, and NDCG@10 only. Coverage and item Gini require per-event
  recommended item IDs and should not be bootstrapped from this file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "A0_fixed_025"
METRICS = ("hit10", "hit50", "ndcg10")
REQUIRED_COLUMNS = {
    "event_id", "method", "target_movie_id", "target_frequency",
    "frequency_group", "target_context_overlap", "context_nonempty",
    "rank_top50", "hit10", "hit50", "ndcg10",
}


def percentile_interval(values, confidence=0.95):
    alpha = 1.0 - confidence
    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1.0 - alpha / 2)),
    ]


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_bootstrap_diff(base, candidate, rng, n_bootstrap):
    """
    Resample shared event indices; calculate candidate minus baseline.
    Returns observed difference, percentile CI, and a two-sided bootstrap
    sign-tail estimate. The latter is descriptive, not a substitute for a
    preregistered inferential test.
    """
    base = np.asarray(base, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if len(base) != len(candidate) or not len(base):
        raise ValueError("Paired metric arrays must have equal nonzero length")

    observed = float(np.mean(candidate - base))
    n = len(base)
    boot_diffs = np.empty(n_bootstrap, dtype=float)
    # Chunk to avoid allocating an unnecessarily large bootstrap index matrix.
    chunk_size = min(250, n_bootstrap)
    completed = 0
    delta = candidate - base
    while completed < n_bootstrap:
        count = min(chunk_size, n_bootstrap - completed)
        idx = rng.integers(0, n, size=(count, n))
        boot_diffs[completed:completed + count] = delta[idx].mean(axis=1)
        completed += count

    ci = percentile_interval(boot_diffs)
    # Two-sided centered bootstrap tail estimate, with a +1 correction.
    centered = boot_diffs - observed
    p_value = (1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (
        n_bootstrap + 1
    )
    return {
        "baseline_mean": float(np.mean(base)),
        "candidate_mean": float(np.mean(candidate)),
        "difference_candidate_minus_baseline": observed,
        "relative_change_percent": (
            float(100.0 * observed / np.mean(base))
            if abs(float(np.mean(base))) > 1e-12 else None
        ),
        "ci95_percentile": ci,
        "bootstrap_two_sided_p_descriptive": float(p_value),
    }


def validate_and_pivot(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("CSV has no data rows")

    for col in ("event_id", "target_movie_id", "target_frequency",
                "target_context_overlap", "context_nonempty",
                "hit10", "hit50", "ndcg10"):
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.duplicated(["event_id", "method"]).any():
        example = df.loc[df.duplicated(["event_id", "method"], keep=False),
                         ["event_id", "method"]].head().to_dict("records")
        raise ValueError(f"Duplicate event_id/method rows detected: {example}")

    methods = list(dict.fromkeys(df["method"].astype(str).tolist()))
    if BASELINE not in methods:
        raise ValueError(f"Baseline method {BASELINE!r} not found. Found: {methods}")

    event_sets = {
        method: set(df.loc[df["method"] == method, "event_id"].tolist())
        for method in methods
    }
    baseline_events = event_sets[BASELINE]
    for method, events in event_sets.items():
        if events != baseline_events:
            raise ValueError(
                f"Method {method!r} does not have exactly the same event IDs "
                f"as baseline (method={len(events)}, baseline={len(baseline_events)})."
            )

    # Confirm event metadata is stable across methods.
    metadata_cols = [
        "target_movie_id", "target_frequency", "frequency_group",
        "target_context_overlap", "context_nonempty",
    ]
    for col in metadata_cols:
        counts = df.groupby("event_id")[col].nunique(dropna=False)
        if (counts > 1).any():
            bad = counts[counts > 1].index[:5].tolist()
            raise ValueError(f"Event metadata column {col!r} differs across methods: {bad}")

    return methods, len(baseline_events)


def subgroup_comparisons(df, method, baseline, rng, n_bootstrap, min_events=30):
    """Paired subgroup comparisons, useful for cold-start diagnostics."""
    subgroups = {
        "zero_shot": df["target_frequency"] == 0,
        "one_shot": df["target_frequency"] == 1,
        "other": df["target_frequency"] > 1,
        "target_context_overlap": df["target_context_overlap"] == 1,
        "target_context_nonoverlap": df["target_context_overlap"] == 0,
        "nonempty_context": df["context_nonempty"] == 1,
        "empty_context": df["context_nonempty"] == 0,
    }
    out = {}
    for label, mask in subgroups.items():
        sub = df.loc[mask]
        b = sub.loc[sub["method"] == baseline].sort_values("event_id")
        c = sub.loc[sub["method"] == method].sort_values("event_id")
        n = len(b)
        if n < min_events:
            out[label] = {"events": n, "status": "too_few_events_for_interval"}
            continue
        entry = {"events": n}
        for metric in METRICS:
            entry[metric] = paired_bootstrap_diff(
                b[metric].to_numpy(), c[metric].to_numpy(), rng, n_bootstrap
            )
        out[label] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results_corrected/adaptive_m2_validation_events.csv",
    )
    parser.add_argument(
        "--out-json",
        default="results_corrected/m2_paired_bootstrap.json",
    )
    parser.add_argument(
        "--out-csv",
        default="results_corrected/m2_paired_bootstrap_summary.csv",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()

    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates; 5000 is recommended.")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)
    methods, n_events = validate_and_pivot(df)
    rng = np.random.default_rng(args.seed)

    # Use event-sorted records to guarantee identical paired ordering.
    indexed = {
        method: df.loc[df["method"] == method].sort_values("event_id").reset_index(drop=True)
        for method in methods
    }
    baseline = indexed[BASELINE]

    results = []
    p_records = {metric: [] for metric in METRICS}
    for method in methods:
        if method == BASELINE:
            for metric in METRICS:
                results.append({
                    "comparison": f"{method} vs {BASELINE}",
                    "method": method,
                    "baseline": BASELINE,
                    "metric": metric,
                    "events": n_events,
                    "baseline_mean": float(baseline[metric].mean()),
                    "candidate_mean": float(baseline[metric].mean()),
                    "difference_candidate_minus_baseline": 0.0,
                    "relative_change_percent": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "bootstrap_two_sided_p_descriptive": 1.0,
                    "holm_adjusted_p_within_metric": 1.0,
                })
            continue

        candidate = indexed[method]
        for metric in METRICS:
            stats = paired_bootstrap_diff(
                baseline[metric].to_numpy(),
                candidate[metric].to_numpy(),
                rng,
                args.bootstrap,
            )
            ci_low, ci_high = stats["ci95_percentile"]
            results.append({
                "comparison": f"{method} vs {BASELINE}",
                "method": method,
                "baseline": BASELINE,
                "metric": metric,
                "events": n_events,
                "baseline_mean": stats["baseline_mean"],
                "candidate_mean": stats["candidate_mean"],
                "difference_candidate_minus_baseline":
                    stats["difference_candidate_minus_baseline"],
                "relative_change_percent": stats["relative_change_percent"],
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "bootstrap_two_sided_p_descriptive":
                    stats["bootstrap_two_sided_p_descriptive"],
                "holm_adjusted_p_within_metric": None,
            })
            p_records[metric].append(
                (len(results) - 1, stats["bootstrap_two_sided_p_descriptive"])
            )

    # Holm adjustment across the six non-baseline comparisons for each metric.
    for metric, entries in p_records.items():
        adjusted = holm_adjust([p for _, p in entries])
        for (result_idx, _), p_adj in zip(entries, adjusted):
            results[result_idx]["holm_adjusted_p_within_metric"] = float(p_adj)

    summary = pd.DataFrame(results)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    # Detailed subgroup results for every non-baseline method.
    subgroup_results = {}
    for method in methods:
        if method == BASELINE:
            continue
        # Independent deterministic stream per method, without affecting the
        # main overall intervals.
        sub_rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in method))
        subgroup_results[method] = subgroup_comparisons(
            df, method, BASELINE, sub_rng, args.bootstrap
        )

    report = {
        "analysis": "paired event-level bootstrap on validation split",
        "input": str(input_path),
        "split": "validation_only",
        "baseline": BASELINE,
        "events_per_method": n_events,
        "methods": methods,
        "bootstrap_replicates": args.bootstrap,
        "random_seed": args.seed,
        "confidence_interval": "95% percentile bootstrap interval for candidate minus baseline",
        "multiple_comparisons": (
            "Holm-adjusted descriptive bootstrap tail estimates across non-baseline "
            "methods separately for each metric. Interpret cautiously; these are not "
            "a replacement for a prespecified inferential test."
        ),
        "overall_results": results,
        "subgroup_results": subgroup_results,
        "limitations": [
            "Bootstrap resamples recommendation events and assumes events are the analysis units; "
            "events from the same conversation may be dependent if multiple events share a conversation.",
            "The CSV does not contain recommended top-10 movie IDs, so catalog coverage and item Gini "
            "cannot be bootstrapped from this file.",
            "Validation results are for method selection and uncertainty analysis only; do not use "
            "these results as final held-out test performance.",
            "A percentile interval crossing zero suggests the observed difference is uncertain under "
            "this bootstrap procedure; it does not prove equivalence."
        ],
        "outputs": {"summary_csv": str(out_csv), "report_json": str(out_json)},
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Methods: {len(methods)} | events per method: {n_events}")
    print(f"Bootstrap replicates: {args.bootstrap} | seed: {args.seed}")
    print("\nOverall paired comparisons (candidate minus A0):")
    display_cols = [
        "method", "metric", "baseline_mean", "candidate_mean",
        "difference_candidate_minus_baseline", "ci95_low", "ci95_high",
        "holm_adjusted_p_within_metric",
    ]
    print(summary[summary["method"] != BASELINE][display_cols].to_string(index=False))
    print(f"\nSaved CSV: {out_csv.resolve()}")
    print(f"Saved JSON: {out_json.resolve()}")


if __name__ == "__main__":
    main()
