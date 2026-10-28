"""Validation-only clustered ranking and fairness uncertainty analysis."""

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
RANK_BOOTSTRAPS = 10000
FAIRNESS_BOOTSTRAPS = 2000
PERMUTATIONS = 20000
SEED = 20261026
BATCH_SIZE = 200
TOL = 1e-10


def holm_adjust(p_values):
    """Holm step-down multiplicity correction."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values for Holm adjustment.")
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def write_csv(path, records):
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def gini_full_catalog(exposures, catalog_size):
    """Gini across the full catalog, including zero-exposure movies."""
    values = np.asarray(exposures, dtype=np.float64)
    if len(values) > catalog_size:
        raise ValueError("Exposed-item count exceeds catalog size.")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Exposure counts must be finite and nonnegative.")
    if catalog_size <= 0 or values.sum() == 0:
        return 0.0

    if len(values) < catalog_size:
        values = np.pad(values, (0, catalog_size - len(values)))
    values.sort()
    n = len(values)
    return float(
        2 * np.dot(np.arange(1, n + 1), values)
        / (n * values.sum()) - (n + 1) / n
    )


def read_inputs():
    for path in (EVENTS, REPORT):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    with EVENTS.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required = {
        "event_id", "conv_id", "method", "target_movie_id",
        *RANK_METRICS, "top10_movie_ids", "top10_group_ids",
    }
    missing = required - columns
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("Event CSV is empty.")

    methods = sorted({r["method"] for r in rows})
    if BASE not in methods or len(methods) < 2:
        raise ValueError("Baseline or candidate methods are missing.")

    events = {}
    movie_to_group = {}

    for row in rows:
        eid, cid, method = (
            row["event_id"], row["conv_id"], row["method"]
        )
        if not eid or not cid or not method:
            raise ValueError("Missing event, conversation, or method ID.")

        event = events.setdefault(
            eid, {"conv_id": cid, "target": None, "by_method": {}}
        )
        if event["conv_id"] != cid:
            raise ValueError(f"Event {eid} maps to multiple conversations.")
        if method in event["by_method"]:
            raise ValueError(f"Duplicate event/method: {eid}/{method}")

        target = int(row["target_movie_id"])
        if event["target"] is not None and event["target"] != target:
            raise ValueError(f"Inconsistent target for event {eid}.")
        event["target"] = target

        for metric in RANK_METRICS:
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {metric}: {eid}/{method}")
            if metric in ("hit10", "hit50") and value not in (0.0, 1.0):
                raise ValueError(f"{metric} must be binary.")
            if metric == "ndcg10" and not 0 <= value <= 1:
                raise ValueError("NDCG@10 must be between zero and one.")

        movies = json.loads(row["top10_movie_ids"])
        groups = json.loads(row["top10_group_ids"])

        if not isinstance(movies, list) or not isinstance(groups, list):
            raise ValueError(f"Top-10 values must be lists: {eid}/{method}")
        if len(movies) != 10 or len(groups) != 10:
            raise ValueError(f"Expected ten movies and groups: {eid}/{method}")
        if any(type(x) is not int for x in movies):
            raise ValueError(f"Movie IDs must be integers: {eid}/{method}")
        if len(set(movies)) != 10:
            raise ValueError(f"Duplicate movie in top 10: {eid}/{method}")
        if any(type(x) is not int or x not in range(GROUP_COUNT) for x in groups):
            raise ValueError(f"Group IDs must be integers 0-4: {eid}/{method}")

        for movie, group in zip(movies, groups):
            old_group = movie_to_group.setdefault(movie, group)
            if old_group != group:
                raise ValueError(
                    f"Movie {movie} has inconsistent group assignments."
                )

        row["_movies"] = movies
        row["_groups"] = groups
        event["by_method"][method] = row

    expected_methods = set(methods)
    for eid, event in events.items():
        if set(event["by_method"]) != expected_methods:
            raise ValueError(f"Incomplete method set for event {eid}.")

    if len(rows) != len(events) * len(methods):
        raise ValueError("Rows != unique events multiplied by methods.")

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("split") != "validation_only":
        raise ValueError("Expected validation-only report.")
    if report.get("test_split_evaluated") not in (False, None):
        raise ValueError("Report does not confirm the test set is untouched.")

    report_results = report.get("results")
    if not isinstance(report_results, list):
        raise ValueError("Validation report has no results list.")
    summary = {
        x["method"]: x for x in report_results if "method" in x
    }
    if set(methods) - set(summary):
        raise ValueError("Some methods are missing from the validation report.")

    catalog_size = int(report["candidate_movies"])
    if catalog_size <= 0 or len(movie_to_group) > catalog_size:
        raise ValueError("Invalid catalog size.")

    by_conv = {}
    for eid, event in events.items():
        by_conv.setdefault(event["conv_id"], []).append(event)

    conv_ids = sorted(by_conv)
    candidates = [m for m in methods if m != BASE]
    n_events = len(events)
    n_conv = len(conv_ids)
    event_counts = np.zeros(n_conv, dtype=np.int64)
    rank_sums = np.zeros(
        (n_conv, len(candidates), len(RANK_METRICS)), dtype=np.float64
    )

    for ci, cid in enumerate(conv_ids):
        for event in by_conv[cid]:
            event_counts[ci] += 1
            base_row = event["by_method"][BASE]
            for cj, candidate in enumerate(candidates):
                candidate_row = event["by_method"][candidate]
                for mi, metric in enumerate(RANK_METRICS):
                    rank_sums[ci, cj, mi] += (
                        float(candidate_row[metric]) - float(base_row[metric])
                    )

    # Sparse conversation-by-movie exposure matrices.
    # Movies never recommended in any top-10 list remain implicit zeros
    # and are included when computing full-catalog Gini.
    exposed_ids = sorted(movie_to_group)
    movie_col = {movie: i for i, movie in enumerate(exposed_ids)}
    conv_col = {cid: i for i, cid in enumerate(conv_ids)}
    matrices, group_counts = {}, {}

    for method in methods:
        rr, cc, vv = [], [], []
        groups_by_conv = np.zeros((n_conv, GROUP_COUNT), dtype=np.int64)

        for row in rows:
            if row["method"] != method:
                continue
            ci = conv_col[row["conv_id"]]
            for movie, group in zip(row["_movies"], row["_groups"]):
                rr.append(ci)
                cc.append(movie_col[movie])
                vv.append(1)
                groups_by_conv[ci, group] += 1

        matrices[method] = coo_matrix(
            (np.asarray(vv, dtype=np.int32), (rr, cc)),
            shape=(n_conv, len(exposed_ids)),
        ).tocsr()
        matrices[method].sum_duplicates()
        group_counts[method] = groups_by_conv

    return (
        rows, methods, candidates, conv_ids, rank_sums, event_counts,
        n_events, catalog_size, exposed_ids, matrices, group_counts, summary,
    )


def ranking_inference(rank_sums, event_counts, rng):
    n_conv, n_candidates, n_metrics = rank_sums.shape
    total_events = int(event_counts.sum())
    observed = rank_sums.sum(axis=0) / total_events

    draws = np.empty(
        (RANK_BOOTSTRAPS, n_candidates, n_metrics), dtype=np.float64
    )
    for start in range(0, RANK_BOOTSTRAPS, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, RANK_BOOTSTRAPS)
        idx = rng.integers(0, n_conv, size=(stop - start, n_conv))
        denom = event_counts[idx].sum(axis=1)
        draws[start:stop] = (
            rank_sums[idx].sum(axis=1) / denom[:, None, None]
        )
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)

    observed_abs = np.abs(rank_sums.sum(axis=0))
    extreme = np.zeros((n_candidates, n_metrics), dtype=np.int64)
    done = 0
    while done < PERMUTATIONS:
        size = min(BATCH_SIZE, PERMUTATIONS - done)
        signs = rng.choice([-1.0, 1.0], size=(size, n_conv, 1, 1))
        permuted = np.abs((signs * rank_sums[None, :, :, :]).sum(axis=1))
        extreme += (permuted >= observed_abs[None, :, :] - TOL).sum(axis=0)
        done += size

    p_raw = (extreme + 1) / (PERMUTATIONS + 1)
    p_all = holm_adjust(p_raw.ravel()).reshape(p_raw.shape)
    p_metric = np.column_stack([
        holm_adjust(p_raw[:, j]) for j in range(n_metrics)
    ])
    return observed, lo, hi, p_raw, p_all, p_metric


def fairness_values(item_counts, group_counts, catalog_size):
    slots = int(group_counts.sum())
    if slots <= 0 or int(item_counts.sum()) != slots:
        raise ValueError("Invalid exposure totals in bootstrap sample.")

    coverage = float(np.count_nonzero(item_counts) / catalog_size)
    gini = gini_full_catalog(item_counts, catalog_size)
    shares = group_counts.astype(np.float64) / slots
    return np.concatenate(([coverage, gini], shares))


def fairness_bootstrap(
    methods, conv_ids, catalog_size, exposed_ids,
    matrices, group_counts, rng,
):
    names = ["catalog_coverage10", "item_gini"] + [
        f"group_{g}_exposure_share" for g in range(GROUP_COUNT)
    ]
    n_methods, n_metrics, n_conv = len(methods), len(names), len(conv_ids)
    point = np.zeros((n_methods, n_metrics), dtype=np.float64)
    draws = np.zeros(
        (FAIRNESS_BOOTSTRAPS, n_methods, n_metrics), dtype=np.float64
    )

    for mi, method in enumerate(methods):
        items = np.asarray(matrices[method].sum(axis=0)).ravel()
        groups = group_counts[method].sum(axis=0)
        point[mi] = fairness_values(items, groups, catalog_size)

    # Use identical sampled conversations for every method so paired
    # candidate-minus-baseline intervals preserve event correspondence.
    for b in range(FAIRNESS_BOOTSTRAPS):
        sampled = rng.integers(0, n_conv, size=n_conv)
        weights = np.bincount(sampled, minlength=n_conv).astype(np.int64)
        for mi, method in enumerate(methods):
            items = np.asarray(weights @ matrices[method]).ravel()
            groups = weights @ group_counts[method]
            draws[b, mi] = fairness_values(items, groups, catalog_size)

    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    base_idx = methods.index(BASE)
    records = []

    for mi, method in enumerate(methods):
        record = {
            "method": method,
            "split": "validation_only",
            "conversation_clusters": n_conv,
            "bootstrap_replicates": FAIRNESS_BOOTSTRAPS,
            "catalog_size": catalog_size,
        }
        for j, name in enumerate(names):
            record[name] = float(point[mi, j])
            record[name + "_ci95_low"] = float(lo[mi, j])
            record[name + "_ci95_high"] = float(hi[mi, j])

            if method == BASE:
                delta = np.zeros(FAIRNESS_BOOTSTRAPS)
                delta_point = 0.0
            else:
                delta = draws[:, mi, j] - draws[:, base_idx, j]
                delta_point = float(point[mi, j] - point[base_idx, j])

            delta_lo, delta_hi = np.quantile(delta, [0.025, 0.975])
            if method != BASE and name in ("catalog_coverage10", "item_gini"):
                mean_delta = float(np.mean(delta))
                median_delta = float(np.median(delta))
                bias = mean_delta - delta_point

                print(
                    f"BOOTSTRAP DIAGNOSTIC {method} {name}: "
                    f"observed={delta_point:+.6f}; "
                    f"mean={mean_delta:+.6f}; "
                    f"median={median_delta:+.6f}; "
                    f"bias={bias:+.6f}; "
                    f"percentile_CI=[{delta_lo:+.6f}, {delta_hi:+.6f}]; "
                    f"basic_CI=[{2 * delta_point - delta_hi:+.6f}, "
                    f"{2 * delta_point - delta_lo:+.6f}]"
                )

            record[name + "_delta_vs_baseline"] = delta_point
            record[name + "_delta_ci95_low"] = float(delta_lo)
            record[name + "_delta_ci95_high"] = float(delta_hi)

        records.append(record)

    return names, point, records


def main():
    (
        rows, methods, candidates, conv_ids, rank_sums, event_counts,
        n_events, catalog_size, exposed_ids, matrices, group_counts, summary,
    ) = read_inputs()

    rng = np.random.default_rng(SEED)
    rank_obs, rank_lo, rank_hi, p_raw, p_all, p_metric = ranking_inference(
        rank_sums, event_counts, rng
    )

    ranking_records = []
    for ci, method in enumerate(candidates):
        for mi, metric in enumerate(RANK_METRICS):
            ranking_records.append({
                "baseline": BASE,
                "method": method,
                "metric": metric,
                "baseline_mean": float(summary[BASE][metric]),
                "candidate_mean": float(summary[method][metric]),
                "difference": float(rank_obs[ci, mi]),
                "difference_percentage_points": float(100 * rank_obs[ci, mi]),
                "cluster_bootstrap_ci95_low": float(rank_lo[ci, mi]),
                "cluster_bootstrap_ci95_high": float(rank_hi[ci, mi]),
                "permutation_p_raw": float(p_raw[ci, mi]),
                "holm_p_all_tests": float(p_all[ci, mi]),
                "holm_p_within_metric": float(p_metric[ci, mi]),
                "conversation_clusters": len(conv_ids),
                "events": n_events,
                "resampling_unit": "conversation",
            })

    fairness_names, fairness_point, fairness_records = fairness_bootstrap(
        methods, conv_ids, catalog_size, exposed_ids, matrices, group_counts, rng
    )

    # Check event-list point estimates against the original validation summary.
    for mi, method in enumerate(methods):
        for j, metric in enumerate(("catalog_coverage10", "item_gini")):
            expected = summary[method].get(metric)
            if expected is not None and not np.isclose(
                float(expected), fairness_point[mi, j], atol=1e-8, rtol=1e-6
            ):
                raise ValueError(
                    f"{metric} mismatch for {method}: event lists="
                    f"{fairness_point[mi, j]:.10f}, report={expected}. "
                    "Ensure the CSV and JSON came from the same run."
                )
        expected_shares = summary[method].get("group_exposure_shares", [])
        actual_shares = fairness_point[mi, 2:].tolist()
        if expected_shares and not np.allclose(
            expected_shares, actual_shares, atol=1e-8, rtol=1e-6
        ):
            raise ValueError(f"Group exposure shares mismatch for {method}.")

    stats_csv = RES / "m2_multiplicity_adjusted_cluster_tests.csv"
    stats_json = RES / "m2_multiplicity_adjusted_cluster_tests.json"
    write_csv(stats_csv, ranking_records)
    stats_json.write_text(json.dumps({
        "split": "validation_only",
        "test_split_evaluated": False,
        "baseline": BASE,
        "events": n_events,
        "conversation_clusters": len(conv_ids),
        "ranking_metrics": list(RANK_METRICS),
        "bootstrap_replicates": RANK_BOOTSTRAPS,
        "permutation_replicates": PERMUTATIONS,
        "seed": SEED,
        "adjustments": {
            "holm_all_tests": f"{len(candidates)} candidates x 3 metrics",
            "holm_within_metric": "Candidates adjusted separately per ranking metric",
        },
        "permutation_assumptions": (
            "Independent conversation clusters and sign-exchangeability under "
            "the null; interpret sign-flip inference cautiously."
        ),
        "results": ranking_records,
    }, indent=2), encoding="utf-8")

    fairness_csv = RES / "m2_validation_fairness_uncertainty.csv"
    fairness_json = RES / "m2_validation_fairness_uncertainty.json"
    write_csv(fairness_csv, fairness_records)
    fairness_json.write_text(json.dumps({
        "split": "validation_only",
        "test_split_evaluated": False,
        "baseline": BASE,
        "catalog_size": catalog_size,
        "events": n_events,
        "conversation_clusters": len(conv_ids),
        "metrics": fairness_names,
        "bootstrap_replicates": FAIRNESS_BOOTSTRAPS,
        "seed": SEED,
        "resampling_unit": "conversation",
        "paired_comparisons": True,
        "gini_definition": "Full candidate catalog, including zero-exposure items.",
        "results": fairness_records,
        "caveat": (
            "Intervals quantify validation-sample uncertainty under conversation "
            "cluster resampling; they do not establish causal fairness or guarantee "
            "generalization to other datasets."
        ),
    }, indent=2), encoding="utf-8")

    fairness_by_method = {r["method"]: r for r in fairness_records}
    tradeoff_records = []
    for method in methods:
        row = {"method": method, "split": "validation_only"}
        for metric in RANK_METRICS:
            row[metric] = summary[method].get(metric)
            row[metric + "_delta_vs_baseline"] = (
                float(summary[method][metric]) - float(summary[BASE][metric])
            )
        for metric in fairness_names:
            f = fairness_by_method[method]
            for suffix in (
                "", "_ci95_low", "_ci95_high", "_delta_vs_baseline",
                "_delta_ci95_low", "_delta_ci95_high",
            ):
                row[metric + suffix] = f[metric + suffix] if suffix else f[metric]
        tradeoff_records.append(row)

    tradeoff_csv = RES / "m2_validation_fairness_accuracy_tradeoff.csv"
    write_csv(tradeoff_csv, tradeoff_records)

    print(
        f"Validation only | conversations: {len(conv_ids)}; events: {n_events}; "
        f"rows: {len(rows)}; catalog: {catalog_size}"
    )
    print("Saved:", stats_csv)
    print("Saved:", stats_json)
    print("Saved:", fairness_csv)
    print("Saved:", fairness_json)
    print("Saved:", tradeoff_csv)

    print("\nRanking comparisons (percentage-point deltas):")
    for r in ranking_records:
        print(
            f"{r['method']:25s} {r['metric']:6s} "
            f"delta={r['difference_percentage_points']:+.3f} pp "
            f"CI=[{100*r['cluster_bootstrap_ci95_low']:+.3f}, "
            f"{100*r['cluster_bootstrap_ci95_high']:+.3f}] pp "
            f"p={r['permutation_p_raw']:.5f} "
            f"Holm-all={r['holm_p_all_tests']:.5f} "
            f"Holm/metric={r['holm_p_within_metric']:.5f}"
        )

    print("\nFairness uncertainty (paired deltas vs baseline):")
    for r in fairness_records:
        print(
            f"{r['method']:25s} "
            f"Coverage={r['catalog_coverage10']:.4f} "
            f"delta={r['catalog_coverage10_delta_vs_baseline']:+.4f} "
            f"CI=[{r['catalog_coverage10_delta_ci95_low']:+.4f}, "
            f"{r['catalog_coverage10_delta_ci95_high']:+.4f}]; "
            f"Gini={r['item_gini']:.4f} "
            f"delta={r['item_gini_delta_vs_baseline']:+.4f} "
            f"CI=[{r['item_gini_delta_ci95_low']:+.4f}, "
            f"{r['item_gini_delta_ci95_high']:+.4f}]"
        )

    print("\nHeld-out test set evaluated: False")


if __name__ == "__main__":
    main()
