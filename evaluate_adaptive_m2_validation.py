"""Validation-only clustered inference for adaptive KGSF + KG rank fusion.

Reads validation event lists and the validation summary only.
Does not train models or evaluate the held-out test set.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_corrected"
EVENTS = RES / "adaptive_m2_validation_events_clustered.csv"
REPORT = RES / "adaptive_m2_validation.json"

BASE = "A0_fixed_025"
RANK_METRICS = ("hit10", "hit50", "ndcg10")
GROUP_COUNT = 5
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 20_000
SEED = 20261026
BATCH_SIZE = 200
TOL = 1e-10


def holm_adjust(p_values):
    """Holm family-wise error correction."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError("Invalid p-value array.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("P-values must be between zero and one.")
    if len(p) == 0:
        return p.copy()

    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    )
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted


def write_csv(path, records):
    if not records:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def exposure_gini(counts, catalog_size):
    """Gini over the entire catalog, including zero-exposure items."""
    values = np.asarray(counts, dtype=np.float64).ravel()

    if catalog_size <= 0 or len(values) > catalog_size:
        raise ValueError("Invalid catalog size.")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid exposure counts.")

    if len(values) < catalog_size:
        values = np.pad(values, (0, catalog_size - len(values)))

    if values.sum() == 0:
        return 0.0

    values.sort()
    n = len(values)
    return float(
        2 * np.dot(np.arange(1, n + 1), values)
        / (n * values.sum()) - (n + 1) / n
    )


def read_inputs():
    """Load and strictly validate the validation-only inputs."""
    for path in (EVENTS, REPORT):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    with EVENTS.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required = {
        "conv_id", "event_id", "method", "target_movie_id",
        *RANK_METRICS, "top10_movie_ids", "top10_group_ids",
    }
    missing = required - columns
    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}. "
            "Regenerate the validation event CSV first."
        )
    if not rows:
        raise ValueError("The validation event CSV is empty.")

    methods = sorted({r["method"] for r in rows})
    if BASE not in methods or len(methods) < 2:
        raise ValueError("Baseline or candidate methods are missing.")

    events = {}
    movie_to_group = {}

    for row in rows:
        eid = row["event_id"]
        cid = row["conv_id"]
        method = row["method"]

        if not eid or not cid or not method:
            raise ValueError("Missing event ID, conversation ID, or method.")

        event = events.setdefault(
            eid, {"conv_id": cid, "target": None, "methods": {}}
        )
        if event["conv_id"] != cid:
            raise ValueError(f"Event {eid} maps to multiple conversations.")
        if method in event["methods"]:
            raise ValueError(f"Duplicate event/method row: {eid}/{method}")

        target = int(row["target_movie_id"])
        if event["target"] is None:
            event["target"] = target
        elif event["target"] != target:
            raise ValueError(f"Inconsistent target for event {eid}.")

        for metric in RANK_METRICS:
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {metric} at {eid}/{method}.")
            if metric in ("hit10", "hit50") and value not in (0, 1):
                raise ValueError(f"{metric} must be binary at {eid}/{method}.")

        movies = json.loads(row["top10_movie_ids"])
        groups = json.loads(row["top10_group_ids"])

        if not isinstance(movies, list) or not isinstance(groups, list):
            raise ValueError(f"Top-10 fields must be lists at {eid}/{method}.")
        if len(movies) != 10 or len(groups) != 10:
            raise ValueError(f"Expected ten items and groups at {eid}/{method}.")
        if any(type(x) is not int for x in movies) or len(set(movies)) != 10:
            raise ValueError(f"Invalid or duplicate movie IDs at {eid}/{method}.")
        if any(type(x) is not int or x not in range(GROUP_COUNT) for x in groups):
            raise ValueError(f"Invalid group IDs at {eid}/{method}.")

        for movie, group in zip(movies, groups):
            previous = movie_to_group.setdefault(movie, group)
            if previous != group:
                raise ValueError(f"Movie {movie} has inconsistent group IDs.")

        row["_movies"] = movies
        row["_groups"] = groups
        event["methods"][method] = row

    expected_methods = set(methods)
    for eid, event in events.items():
        if set(event["methods"]) != expected_methods:
            raise ValueError(f"Incomplete method set for event {eid}.")

    if len(rows) != len(events) * len(methods):
        raise ValueError("Row count does not equal events times methods.")

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("split") != "validation_only":
        raise ValueError("Report is not explicitly validation-only.")
    if report.get("test_split_evaluated") is not False:
        raise ValueError("Test status is not explicitly marked False.")

    summaries = report.get("results")
    if not isinstance(summaries, list):
        raise ValueError("Validation summary has no results list.")

    summary_by_method = {
        r["method"]: r for r in summaries if "method" in r
    }
    if set(methods) - set(summary_by_method):
        raise ValueError("Event methods and summary methods do not match.")

    catalog_size = int(report["candidate_movies"])
    if catalog_size <= 0 or len(movie_to_group) > catalog_size:
        raise ValueError("Invalid candidate catalog size.")

    by_conversation = {}
    for eid, event in events.items():
        by_conversation.setdefault(event["conv_id"], []).append((eid, event))

    conv_ids = sorted(by_conversation)
    candidates = [m for m in methods if m != BASE]
    n_events = len(events)

    # Sum paired candidate-minus-baseline ranking differences by conversation.
    rank_sums = np.zeros(
        (len(conv_ids), len(candidates), len(RANK_METRICS)),
        dtype=np.float64,
    )
    event_counts = np.zeros(len(conv_ids), dtype=np.int64)

    for ci, cid in enumerate(conv_ids):
        for eid, event in by_conversation[cid]:
            event_counts[ci] += 1
            base_row = event["methods"][BASE]
            for cj, candidate in enumerate(candidates):
                candidate_row = event["methods"][candidate]
                for mi, metric in enumerate(RANK_METRICS):
                    rank_sums[ci, cj, mi] += (
                        float(candidate_row[metric]) - float(base_row[metric])
                    )

    # Sparse conversation-by-item exposure matrices.
    exposed_movies = sorted(movie_to_group)
    movie_col = {movie: i for i, movie in enumerate(exposed_movies)}
    conv_index = {cid: i for i, cid in enumerate(conv_ids)}
    item_matrices = {}
    group_matrices = {}

    for method in methods:
        rr, cc, vv = [], [], []
        per_conv_groups = np.zeros(
            (len(conv_ids), GROUP_COUNT), dtype=np.int64
        )

        for row in rows:
            if row["method"] != method:
                continue
            ci = conv_index[row["conv_id"]]
            for movie, group in zip(row["_movies"], row["_groups"]):
                rr.append(ci)
                cc.append(movie_col[movie])
                vv.append(1)
                per_conv_groups[ci, group] += 1

        matrix = coo_matrix(
            (np.asarray(vv, dtype=np.int32), (rr, cc)),
            shape=(len(conv_ids), len(exposed_movies)),
            dtype=np.int32,
        ).tocsr()
        matrix.sum_duplicates()
        item_matrices[method] = matrix
        group_matrices[method] = per_conv_groups

    return (
        rows, methods, candidates, conv_ids, rank_sums, event_counts,
        n_events, catalog_size, exposed_movies, item_matrices,
        group_matrices, summary_by_method,
    )


def ranking_inference(rank_sums, event_counts, rng):
    """Conversation-cluster bootstrap CIs and sign-flip tests."""
    n_conv, n_candidates, n_metrics = rank_sums.shape
    total_events = int(event_counts.sum())
    observed = rank_sums.sum(axis=0) / total_events

    draws = np.empty(
        (BOOTSTRAP_REPLICATES, n_candidates, n_metrics), dtype=np.float64
    )
    for start in range(0, BOOTSTRAP_REPLICATES, BATCH_SIZE):
        stop = min(BOOTSTRAP_REPLICATES, start + BATCH_SIZE)
        idx = rng.integers(0, n_conv, size=(stop - start, n_conv))
        denominator = event_counts[idx].sum(axis=1)
        draws[start:stop] = (
            rank_sums[idx].sum(axis=1) / denominator[:, None, None]
        )

    # Basic (centered) bootstrap intervals.
    # Center the bootstrap errors on the observed full-sample estimates.
    bootstrap_errors = draws - point[None, :, :]
    error_lo, error_hi = np.quantile(
    bootstrap_errors, [0.025, 0.975], axis=0
    )

    lo = point - error_hi
    hi = point - error_lo

    observed_abs = np.abs(rank_sums.sum(axis=0))
    extreme = np.zeros((n_candidates, n_metrics), dtype=np.int64)
    done = 0

    while done < PERMUTATION_REPLICATES:
        batch_n = min(BATCH_SIZE, PERMUTATION_REPLICATES - done)
        signs = rng.choice([-1.0, 1.0], size=(batch_n, n_conv, 1, 1))
        permuted = np.abs((signs * rank_sums[None, :, :, :]).sum(axis=1))
        extreme += (permuted >= observed_abs[None, :, :] - TOL).sum(axis=0)
        done += batch_n

    p_raw = (extreme + 1) / (PERMUTATION_REPLICATES + 1)
    p_all = holm_adjust(p_raw.ravel()).reshape(p_raw.shape)
    p_metric = np.column_stack([
        holm_adjust(p_raw[:, i]) for i in range(n_metrics)
    ])

    return observed, low, high, p_raw, p_all, p_metric


def fairness_vector(item_counts, group_counts, catalog_size):
    """Return coverage, full-catalog Gini and five group exposure shares."""
    slots = int(group_counts.sum())
    if slots <= 0 or int(item_counts.sum()) != slots:
        raise ValueError("Inconsistent or empty exposure counts.")

    coverage = np.count_nonzero(item_counts) / catalog_size
    gini = exposure_gini(item_counts, catalog_size)
    shares = group_counts.astype(np.float64) / slots
    return np.concatenate(([coverage, gini], shares))


def fairness_bootstrap(
    methods, conv_ids, catalog_size, exposed_movies,
    item_matrices, group_matrices, rng,
):
    """Paired conversation-cluster bootstrap for fairness metrics."""
    names = ["catalog_coverage10", "item_gini"] + [
        f"group_{g}_exposure_share" for g in range(GROUP_COUNT)
    ]
    n_conv = len(conv_ids)
    n_methods = len(methods)
    n_metrics = len(names)

    point = np.zeros((n_methods, n_metrics), dtype=np.float64)
    draws = np.zeros(
        (BOOTSTRAP_REPLICATES, n_methods, n_metrics), dtype=np.float64
    )

    for mi, method in enumerate(methods):
        item_counts = np.asarray(item_matrices[method].sum(axis=0)).ravel()
        group_counts = group_matrices[method].sum(axis=0)
        point[mi] = fairness_vector(item_counts, group_counts, catalog_size)

    # Identical conversation samples across methods preserve paired comparisons.
    for start in range(0, BOOTSTRAP_REPLICATES, BATCH_SIZE):
        stop = min(BOOTSTRAP_REPLICATES, start + BATCH_SIZE)
        for draw in range(start, stop):
            sampled = rng.integers(0, n_conv, size=n_conv)
            weights = np.bincount(sampled, minlength=n_conv)
            for mi, method in enumerate(methods):
                items = np.asarray(
                    weights @ item_matrices[method]
                ).ravel()
                groups = weights @ group_matrices[method]
                draws[draw, mi] = fairness_vector(
                    items, groups, catalog_size
                )

    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    base_i = methods.index(BASE)
    records = []

    for mi, method in enumerate(methods):
        record = {
            "method": method,
            "split": "validation_only",
            "conversation_clusters": n_conv,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "catalog_size": catalog_size,
        }
        for ki, name in enumerate(names):
            record[name] = float(point[mi, ki])
            record[name + "_ci95_low"] = float(low[mi, ki])
            record[name + "_ci95_high"] = float(high[mi, ki])

            delta = draws[:, mi, ki] - draws[:, base_i, ki]
            delta_point = point[mi, ki] - point[base_i, ki]
            delta_errors = delta - delta_point
            delta_error_lo, delta_error_hi = np.quantile(
            delta_errors, [0.025, 0.975]
            )

            delta_low = delta_point - delta_error_hi
            delta_high = delta_point - delta_error_lo

            record[name + "_delta_vs_baseline"] = float(delta_point)
            record[name + "_delta_ci95_low"] = (
                0.0 if method == BASE else float(delta_low)
            )
            record[name + "_delta_ci95_high"] = (
                0.0 if method == BASE else float(delta_high)
            )

        records.append(record)

    return names, point, draws, records


def main():
    (
        rows, methods, candidates, conv_ids, rank_sums, event_counts,
        n_events, catalog_size, exposed_movies, item_matrices,
        group_matrices, summary_by_method,
    ) = read_inputs()

    rng = np.random.default_rng(SEED)
    observed, rank_low, rank_high, p_raw, p_all, p_metric = (
        ranking_inference(rank_sums, event_counts, rng)
    )

    ranking_records = []
    for ci, method in enumerate(candidates):
        for mi, metric in enumerate(RANK_METRICS):
            ranking_records.append({
                "baseline": BASE,
                "method": method,
                "metric": metric,
                "baseline_mean": float(summary_by_method[BASE][metric]),
                "candidate_mean": float(summary_by_method[method][metric]),
                "difference": float(observed[ci, mi]),
                "difference_percentage_points": float(100 * observed[ci, mi]),
                "cluster_bootstrap_ci95_low": float(rank_low[ci, mi]),
                "cluster_bootstrap_ci95_high": float(rank_high[ci, mi]),
                "permutation_p_raw": float(p_raw[ci, mi]),
                "holm_p_all_tests": float(p_all[ci, mi]),
                "holm_p_within_metric": float(p_metric[ci, mi]),
                "conversation_clusters": len(conv_ids),
                "events": n_events,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "permutation_replicates": PERMUTATION_REPLICATES,
                "seed": SEED,
                "resampling_unit": "conversation",
            })

    (
        fairness_names, fairness_point, fairness_draws, fairness_records,
    ) = fairness_bootstrap(
        methods, conv_ids, catalog_size, exposed_movies,
        item_matrices, group_matrices, rng,
    )

    # Check the event-list calculations against the original report.
    for mi, method in enumerate(methods):
        summary = summary_by_method[method]
        for ki, metric in enumerate(("catalog_coverage10", "item_gini")):
            expected = summary.get(metric)
            actual = fairness_point[mi, ki]
            if expected is not None and not np.isclose(
                float(expected), float(actual), atol=1e-8, rtol=1e-6
            ):
                raise ValueError(
                    f"{metric} mismatch for {method}: event lists={actual:.8f}, "
                    f"summary={float(expected):.8f}. Ensure both inputs "
                    "were generated by the same validation run."
                )

        expected_shares = summary.get("group_exposure_shares", [])
        actual_shares = fairness_point[mi, 2:].tolist()
        if expected_shares and (
            len(expected_shares) != GROUP_COUNT
            or not np.allclose(expected_shares, actual_shares, atol=1e-8, rtol=1e-6)
        ):
            raise ValueError(f"Group exposure mismatch for {method}.")

    stats_csv = RES / "m2_multiplicity_adjusted_cluster_tests.csv"
    stats_json = RES / "m2_multiplicity_adjusted_cluster_tests.json"
    fairness_csv = RES / "m2_validation_fairness_uncertainty.csv"
    fairness_json = RES / "m2_validation_fairness_uncertainty.json"
    tradeoff_csv = RES / "m2_validation_fairness_accuracy_tradeoff.csv"

    write_csv(stats_csv, ranking_records)
    stats_json.write_text(json.dumps({
        "split": "validation_only",
        "test_split_evaluated": False,
        "baseline": BASE,
        "events": n_events,
        "conversation_clusters": len(conv_ids),
        "methods": methods,
        "ranking_metrics": list(RANK_METRICS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "seed": SEED,
        "adjustments": {
            "holm_p_all_tests": "all candidate-by-ranking-metric comparisons",
            "holm_p_within_metric": "candidate comparisons separately per metric",
        },
        "permutation_assumptions": (
            "Independent conversation clusters and sign-exchangeability of "
            "conversation-level paired differences under the null."
        ),
        "results": ranking_records,
    }, indent=2), encoding="utf-8")

    write_csv(fairness_csv, fairness_records)
    fairness_json.write_text(json.dumps({
        "split": "validation_only",
        "test_split_evaluated": False,
        "baseline": BASE,
        "catalog_size": catalog_size,
        "events": n_events,
        "conversation_clusters": len(conv_ids),
        "metrics": fairness_names,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "resampling_unit": "conversation",
        "paired_comparisons": True,
        "gini_includes_zero_exposure_catalog_items": True,
        "results": fairness_records,
        "caveat": (
            "Intervals describe validation-sample uncertainty under "
            "conversation-cluster resampling; they do not prove causal fairness "
            "or guarantee generalization."
        ),
    }, indent=2), encoding="utf-8")

    fair_by_method = {r["method"]: r for r in fairness_records}
    tradeoff = []
    for method in methods:
        summary = summary_by_method[method]
        fair = fair_by_method[method]
        record = {"method": method, "split": "validation_only"}

        for metric in RANK_METRICS:
            record[metric] = summary.get(metric)
            record[metric + "_delta_vs_baseline"] = (
                float(summary[metric]) - float(summary_by_method[BASE][metric])
            )

        for metric in fairness_names:
            for suffix in (
                "", "_ci95_low", "_ci95_high",
                "_delta_vs_baseline", "_delta_ci95_low", "_delta_ci95_high",
            ):
                record[metric + suffix] = fair[metric + suffix]

        tradeoff.append(record)

    write_csv(tradeoff_csv, tradeoff)

    print(
        f"Validation only | conversations: {len(conv_ids)}; "
        f"events: {n_events}; rows: {len(rows)}; catalog: {catalog_size}"
    )
    print("Saved ranking inference:", stats_csv)
    print("Saved ranking JSON:", stats_json)
    print("Saved fairness uncertainty CSV:", fairness_csv)
    print("Saved fairness uncertainty JSON:", fairness_json)
    print("Saved updated trade-off table:", tradeoff_csv)

    print("\nCluster-adjusted ranking comparisons (percentage-point deltas):")
    for result in ranking_records:
        print(
            f"{result['method']:25s} {result['metric']:6s} "
            f"delta={result['difference_percentage_points']:+.3f} pp "
            f"CI=[{100*result['cluster_bootstrap_ci95_low']:+.3f}, "
            f"{100*result['cluster_bootstrap_ci95_high']:+.3f}] pp "
            f"p={result['permutation_p_raw']:.5f} "
            f"Holm-all={result['holm_p_all_tests']:.5f} "
            f"Holm/metric={result['holm_p_within_metric']:.5f}"
        )

    print("\nFairness uncertainty (validation; paired deltas):")
    for result in fairness_records:
        print(
            f"{result['method']:25s} "
            f"Coverage={result['catalog_coverage10']:.4f} "
            f"CI=[{result['catalog_coverage10_ci95_low']:.4f}, "
            f"{result['catalog_coverage10_ci95_high']:.4f}] "
            f"delta={result['catalog_coverage10_delta_vs_baseline']:+.4f} "
            f"delta-CI=[{result['catalog_coverage10_delta_ci95_low']:+.4f}, "
            f"{result['catalog_coverage10_delta_ci95_high']:+.4f}]; "
            f"Gini={result['item_gini']:.4f} "
            f"CI=[{result['item_gini_ci95_low']:.4f}, "
            f"{result['item_gini_ci95_high']:.4f}] "
            f"delta={result['item_gini_delta_vs_baseline']:+.4f} "
            f"delta-CI=[{result['item_gini_delta_ci95_low']:+.4f}, "
            f"{result['item_gini_delta_ci95_high']:+.4f}]"
        )


if __name__ == "__main__":
    main()