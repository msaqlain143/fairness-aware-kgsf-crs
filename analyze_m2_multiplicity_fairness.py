"""Validation-only clustered inference and fairness/accuracy trade-off.

Reads results_corrected/adaptive_m2_validation_events_clustered.csv and
results_corrected/adaptive_m2_validation.json. Does not evaluate test data.
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_corrected"
EVENTS = RES / "adaptive_m2_validation_events_clustered.csv"
REPORT = RES / "adaptive_m2_validation.json"
BASE = "A0_fixed_025"
METRICS = ("hit10", "hit50", "ndcg10")
B = 10000
P = 20000
SEED = 20261026


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    )
    result = np.empty_like(p)
    result[order] = adj_sorted
    return result


def read_data():
    with EVENTS.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty input: {EVENTS}")
    required = {"conv_id", "event_id", "method", *METRICS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    methods = sorted({r["method"] for r in rows})
    if BASE not in methods:
        raise ValueError(f"Baseline {BASE} missing")
    ev = {}
    for r in rows:
        eid, cid, method = r["event_id"], r["conv_id"], r["method"]
        if not eid or not cid:
            raise ValueError("Missing event_id or conv_id")
        e = ev.setdefault(eid, {"conv": cid, "by_method": {}})
        if e["conv"] != cid:
            raise ValueError(f"Event {eid} links to multiple conversations")
        if method in e["by_method"]:
            raise ValueError(f"Duplicate event/method: {eid}/{method}")
        e["by_method"][method] = r

    expected = set(methods)
    for eid, e in ev.items():
        if set(e["by_method"]) != expected:
            raise ValueError(f"Incomplete method set for event {eid}")

    by_conv = {}
    for e in ev.values():
        by_conv.setdefault(e["conv"], []).append(e)
    conv_ids = sorted(by_conv)
    candidates = [m for m in methods if m != BASE]
    # One paired candidate-minus-baseline sum per conversation/candidate/metric.
    sums = np.zeros((len(conv_ids), len(candidates), len(METRICS)))
    counts = np.zeros(len(conv_ids), dtype=np.int64)
    for ci, cid in enumerate(conv_ids):
        for e in by_conv[cid]:
            counts[ci] += 1
            b = e["by_method"][BASE]
            for cj, m in enumerate(candidates):
                c = e["by_method"][m]
                for ki, metric in enumerate(METRICS):
                    d = float(c[metric]) - float(b[metric])
                    if not np.isfinite(d):
                        raise ValueError(f"Non-finite value for {eid}/{metric}")
                    sums[ci, cj, ki] += d
    if len(rows) != len(ev) * len(methods):
        raise ValueError("Rows != unique events x methods")
    return rows, methods, candidates, conv_ids, sums, counts, len(ev)


def main():
    rows, methods, candidates, conv_ids, sums, counts, n_events = read_data()
    rng = np.random.default_rng(SEED)
    nconv = len(conv_ids)
    denom = int(counts.sum())
    observed = sums.sum(axis=0) / denom

    # Conversation-cluster bootstrap percentile intervals.
    draws = np.empty((B, len(candidates), len(METRICS)))
    for start in range(0, B, 200):
        stop = min(B, start + 200)
        idx = rng.integers(0, nconv, size=(stop-start, nconv))
        den = counts[idx].sum(axis=1)
        draws[start:stop] = sums[idx].sum(axis=1) / den[:, None, None]
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)

    # Paired conversation-level sign-flip permutation test.
    # The whole conversation contribution is flipped, never individual events.
    obs_sum = np.abs(sums.sum(axis=0))
    extreme = np.zeros_like(obs_sum, dtype=np.int64)
    done = 0
    while done < P:
        n = min(200, P-done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, nconv, 1, 1))
        perm = (signs * sums[None, :, :, :]).sum(axis=1)
        extreme += (np.abs(perm) >= obs_sum[None, :, :] - 1e-12).sum(axis=0)
        done += n
    p_raw = (extreme + 1) / (P + 1)

    # Two adjustment families: all 18 tests, and six comparisons within each metric.
    p_all = holm(p_raw.ravel()).reshape(p_raw.shape)
    p_within = np.column_stack([holm(p_raw[:, k]) for k in range(len(METRICS))])

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("split") != "validation_only":
        raise ValueError("Expected validation-only JSON")
    rep = {r["method"]: r for r in report["results"]}
    out = []
    for ci, method in enumerate(candidates):
        for ki, metric in enumerate(METRICS):
            out.append({
                "baseline": BASE, "method": method, "metric": metric,
                "baseline_mean": rep[BASE][metric],
                "candidate_mean": rep[method][metric],
                "difference": float(observed[ci, ki]),
                "difference_percentage_points": float(100*observed[ci, ki]),
                "cluster_bootstrap_ci95_low": float(lo[ci, ki]),
                "cluster_bootstrap_ci95_high": float(hi[ci, ki]),
                "permutation_p_raw": float(p_raw[ci, ki]),
                "holm_p_all_18_tests": float(p_all[ci, ki]),
                "holm_p_within_metric_6_tests": float(p_within[ci, ki]),
                "conversation_clusters": nconv, "events": n_events,
                "bootstrap_replicates": B, "permutation_replicates": P,
                "seed": SEED, "resampling_unit": "conversation",
            })

    def write_csv(path, data):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    stats_csv = RES / "m2_multiplicity_adjusted_cluster_tests.csv"
    stats_json = RES / "m2_multiplicity_adjusted_cluster_tests.json"
    write_csv(stats_csv, out)
    stats_json.write_text(json.dumps({
        "split": "validation_only", "baseline": BASE, "events": n_events,
        "conversation_clusters": nconv, "methods": methods,
        "metrics": list(METRICS), "bootstrap_replicates": B,
        "permutation_replicates": P, "seed": SEED,
        "adjustments": {
            "holm_p_all_18_tests": "six candidates x three ranking metrics",
            "holm_p_within_metric_6_tests": "six candidates adjusted separately per metric",
        },
        "test_split_evaluated": False, "results": out,
        "caveat": (
            "Exploratory validation analysis. Cluster sign-flip tests assume "
            "independent conversation clusters and sign exchangeability under the null. "
            "Fairness metrics below are descriptive; the event CSV lacks item-level top-10 "
            "exposures needed to bootstrap fairness metrics."
        )
    }, indent=2), encoding="utf-8")

    # Descriptive fairness-accuracy table from the existing validation JSON.
    keys = ("hit10", "hit50", "ndcg10", "catalog_coverage10", "item_gini")
    fairness = []
    for method in [BASE] + candidates:
        r = rep[method]
        base = rep[BASE]
        item = {"method": method}
        for key in keys:
            item[key] = r.get(key)
            item[key + "_delta"] = 0.0 if method == BASE else (
                None if r.get(key) is None or base.get(key) is None
                else r[key] - base[key]
            )
        shares = r.get("group_exposure_shares", [])
        base_shares = base.get("group_exposure_shares", [])
        item["group_exposure_shares"] = json.dumps(shares)
        item["max_group_exposure_share"] = max(shares) if shares else None
        item["max_group_exposure_share_delta"] = (
            max(shares)-max(base_shares) if shares and base_shares else None
        )
        fairness.append(item)
    trade_csv = RES / "m2_validation_fairness_accuracy_tradeoff.csv"
    write_csv(trade_csv, fairness)

    print(f"Conversations: {nconv}; events: {n_events}; rows: {len(rows)}")
    print(f"Saved: {stats_csv}")
    print(f"Saved: {stats_json}")
    print(f"Saved: {trade_csv}")
    print("\nAdjusted ranking comparisons (delta in percentage points):")
    for r in out:
        print(
            f"{r['method']:25s} {r['metric']:6s} "
            f"delta={r['difference_percentage_points']:+.3f} pp "
            f"CI=[{100*r['cluster_bootstrap_ci95_low']:+.3f},"
            f" {100*r['cluster_bootstrap_ci95_high']:+.3f}] pp "
            f"p={r['permutation_p_raw']:.5f} "
            f"Holm18={r['holm_p_all_18_tests']:.5f} "
            f"Holm/metric={r['holm_p_within_metric_6_tests']:.5f}"
        )
    print("\nFairness/accuracy trade-off (descriptive validation metrics):")
    for r in fairness:
        print(
            f"{r['method']:25s} Hit10={r['hit10']:.4f} "
            f"({100*r['hit10_delta']:+.3f} pp); "
            f"Hit50={r['hit50']:.4f} ({100*r['hit50_delta']:+.3f} pp); "
            f"NDCG10={r['ndcg10']:.4f} ({100*r['ndcg10_delta']:+.3f} pp); "
            f"Coverage={r['catalog_coverage10']:.4f} "
            f"({100*r['catalog_coverage10_delta']:+.3f} pp); "
            f"Gini={r['item_gini']:.4f} ({r['item_gini_delta']:+.5f}); "
            f"max group-share delta={r['max_group_exposure_share_delta']:+.5f}"
        )


if __name__ == "__main__":
    main()
