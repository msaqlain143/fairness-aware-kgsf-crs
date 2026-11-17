#!/usr/bin/env python
"""
Experiment 22.24C — Paired cluster inference validation.

Diagnostic-only. Uses the already-saved 22.24B event rankings and the
original 22.24C bootstrap archive. It does NOT rerun KGSF, retrieval,
the ranking policy, or the independent-test evaluation.

Important inferential distinction:
- Conversation-cluster resampling is conditional on the frozen outcomes.
- 22.24B updates a global exposure ledger sequentially across event_id.
  This creates cross-conversation policy dependence/interference.
- Therefore the cluster bootstrap and paired cluster label-swap test below
  are diagnostics for the frozen outcomes, not automatically valid
  policy-level inference. The label-swap test is explicitly marked
  exploratory because treatment labels were not randomized and clusters
  may not be exchangeable under the sequential policy.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path("results_final_test_m2/experiment_22_24B")
DEFAULT_KBRD = Path(
    "results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv"
)
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_PROFILES = Path(
    "results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"
)
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_INFERENCE = Path("22_24C_independent_test_statistical_inference.py")
DEFAULT_22_24B_SCRIPT = Path("22_24B_full_framework_independent_test.py")
DEFAULT_GATE_SCRIPT = Path("22_18_contextual_relevance_gate_validation.py")
DEFAULT_ARCHIVE = Path(
    "results_final_test_m2/experiment_22_24C/bootstrap_distributions.npz"
)
DEFAULT_OUT = Path(
    "results_final_test_m2/experiment_22_24C/paired_cluster_validation"
)
N_BOOT = 10000
BOOT_SEED = 20261115
PERM_SEED = 20261117
EPS = 1e-12
MIN_CELL_CONVERSATIONS = 20


def load_module(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def cluster_sum(values, event_cluster_index, n_clusters):
    out = np.zeros((n_clusters,) + values.shape[1:], dtype=np.float64)
    np.add.at(out, event_cluster_index, values)
    return out


def mad_from_totals(exposure, opportunity, mask):
    exposure = np.asarray(exposure, dtype=np.float64)
    opportunity = np.asarray(opportunity, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if np.any(opportunity[mask] <= EPS):
        raise ValueError("A fixed fairness cell has zero opportunity.")
    log_rate = np.log(
        (exposure[mask] + EPS) / (opportunity[mask] + EPS)
    )
    return float(np.mean(np.abs(log_rate - np.mean(log_rate))))


def bca_interval(bootstrap, observed, jackknife, alpha=0.05):
    """BCa interval from bootstrap draws and delete-one-cluster jackknife."""
    boot = np.asarray(bootstrap, dtype=float)
    jack = np.asarray(jackknife, dtype=float)
    boot = boot[np.isfinite(boot)]
    jack = jack[np.isfinite(jack)]
    if len(boot) < 1000 or len(jack) < 20:
        return {"available": False, "reason": "Insufficient bootstrap/jackknife draws"}

    # Mid-rank correction avoids infinite z0 when no draw is below/equal.
    prop = (np.count_nonzero(boot < observed) +
            0.5 * np.count_nonzero(boot == observed)) / len(boot)
    prop = min(max(prop, 1.0 / (2 * len(boot))), 1.0 - 1.0 / (2 * len(boot)))
    z0 = NormalDist().inv_cdf(prop)

    jbar = float(np.mean(jack))
    dif = jbar - jack
    denom = 6.0 * (float(np.sum(dif ** 2)) ** 1.5)
    accel = float(np.sum(dif ** 3) / denom) if denom > 0 else 0.0

    adjusted = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = NormalDist().inv_cdf(q)
        denominator = 1.0 - accel * (z0 + zq)
        if abs(denominator) < 1e-12:
            return {
                "available": False, "reason": "BCa transformation singular",
                "z0": z0, "acceleration": accel,
            }
        adjusted_q = NormalDist().cdf(
            z0 + (z0 + zq) / denominator
        )
        adjusted.append(float(np.clip(adjusted_q, 0.0, 1.0)))

    lo, hi = np.quantile(boot, adjusted)
    return {
        "available": True,
        "low": float(lo),
        "high": float(hi),
        "z0": float(z0),
        "acceleration": float(accel),
        "adjusted_quantiles": adjusted,
    }


def distribution_summary(name, values, observed, jackknife):
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    centered = x - mean
    skew = (
        float(np.mean(centered ** 3) / (np.mean(centered ** 2) ** 1.5))
        if np.mean(centered ** 2) > 0 else 0.0
    )
    q025, q975 = np.quantile(x, [0.025, 0.975])
    basic_low = 2 * observed - q975
    basic_high = 2 * observed - q025
    bca = bca_interval(x, observed, jackknife)
    return {
        "metric": name,
        "n_bootstrap": int(len(x)),
        "observed_delta": float(observed),
        "bootstrap_mean": mean,
        "bootstrap_median": float(np.median(x)),
        "bootstrap_bias_mean_minus_observed": mean - observed,
        "bootstrap_sd": sd,
        "bootstrap_skewness": skew,
        "fraction_below_zero": float(np.mean(x < 0)),
        "fraction_above_zero": float(np.mean(x > 0)),
        "percentile_ci95_low": float(q025),
        "percentile_ci95_high": float(q975),
        "basic_ci95_low": float(basic_low),
        "basic_ci95_high": float(basic_high),
        "bca": bca,
        "q001": float(np.quantile(x, 0.001)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "q999": float(np.quantile(x, 0.999)),
    }


def inspect_sequential_design(script_22_24b: Path, gate_script: Path):
    source_b = script_22_24b.read_text(encoding="utf-8")
    source_gate = gate_script.read_text(encoding="utf-8")
    checks = {
        "events_iterated_in_sorted_event_id_order":
            bool(re.search(r"for\s+event_id\s+in\s+sorted\(events\)", source_b)),
        "proposed_ledger_created_once_before_event_loop":
            bool(re.search(r"proposed_ledger\s*=\s*ref\.ExposureLedger\(\)", source_b)),
        "contextual_reranker_receives_proposed_ledger":
            bool(re.search(
                r"contextual_gate_rerank\([\s\S]{0,500}?proposed_ledger",
                source_b
            )),
        "proposed_ledger_updated_inside_event_loop":
            bool(re.search(
                r"for\s+event_id\s+in\s+sorted\(events\):[\s\S]{0,5000}?"
                r"exposure_update\(\s*proposed_ledger",
                source_b
            )),
        "gate_source_mentions_ledger_state":
            "ledger" in source_gate.lower(),
    }

    # AST inspection identifies whether the reranker signature/body uses a
    # ledger argument or ledger attribute. This is static evidence, not runtime
    # proof of every state transition.
    try:
        tree = ast.parse(source_gate)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name == "contextual_gate_rerank":
                target = node
                break
        if target is not None:
            segment = ast.get_source_segment(source_gate, target) or ""
            checks["contextual_gate_rerank_has_ledger_parameter"] = (
                "ledger" in [arg.arg for arg in target.args.args]
            )
            checks["contextual_gate_rerank_body_references_ledger"] = (
                "ledger" in segment.lower()
            )
        else:
            checks["contextual_gate_rerank_found"] = False
    except SyntaxError:
        checks["gate_ast_parse_pass"] = False

    sequential = bool(
        checks.get("events_iterated_in_sorted_event_id_order")
        and checks.get("proposed_ledger_created_once_before_event_loop")
        and checks.get("contextual_reranker_receives_proposed_ledger")
        and checks.get("proposed_ledger_updated_inside_event_loop")
    )
    return {
        "checks": checks,
        "sequential_global_ledger_detected": sequential,
        "interpretation": (
            "The proposed policy is stateful across the sorted event stream. "
            "A conversation's frozen proposed outcomes may depend on preceding "
            "events from other conversations. Ordinary independent-cluster "
            "bootstrap/permutation assumptions therefore do not automatically "
            "hold for policy-level inference."
            if sequential else
            "Static checks did not establish the complete sequential dependency; "
            "inspect the gate implementation before making an independence claim."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    ap.add_argument("--catalog-pickle", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--category-profiles", type=Path, default=DEFAULT_PROFILES)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--inference-script", type=Path, default=DEFAULT_INFERENCE)
    ap.add_argument("--script-22-24b", type=Path, default=DEFAULT_22_24B_SCRIPT)
    ap.add_argument("--gate-script", type=Path, default=DEFAULT_GATE_SCRIPT)
    ap.add_argument("--bootstrap-archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--replicates", type=int, default=N_BOOT)
    ap.add_argument("--minimum-cell-conversations", type=int,
                    default=MIN_CELL_CONVERSATIONS)
    args = ap.parse_args()

    required_paths = [
        args.results_dir / "baseline_event_results.csv",
        args.results_dir / "proposed_event_results.csv",
        args.kbrd, args.catalog_pickle, args.category_profiles,
        args.reference, args.inference_script, args.script_22_24b,
        args.gate_script, args.bootstrap_archive,
    ]
    missing = [str(p) for p in required_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing:\n" + "\n".join(missing))
    if args.replicates < 1000:
        raise ValueError("--replicates must be at least 1000.")

    print("=" * 78)
    print("EXPERIMENT 22.24C — PAIRED CLUSTER INFERENCE VALIDATION")
    print("=" * 78)

    inference = load_module(args.inference_script, "exp_22_24c_reference")
    ref = inference.load_reference(args.reference)
    item2group, catalog, categories = ref.load_metadata(
        args.catalog_pickle, args.category_profiles
    )

    baseline = pd.read_csv(args.results_dir / "baseline_event_results.csv")
    proposed = pd.read_csv(args.results_dir / "proposed_event_results.csv")
    kbrd = pd.read_csv(args.kbrd)

    for frame_name, frame in (("baseline", baseline), ("proposed", proposed),
                              ("KBRD", kbrd)):
        if frame["event_id"].duplicated().any():
            raise ValueError(f"Duplicate event_id in {frame_name}.")
    baseline = baseline.sort_values("event_id").reset_index(drop=True)
    proposed = proposed.sort_values("event_id").reset_index(drop=True)
    kbrd = kbrd.sort_values("event_id").reset_index(drop=True)

    if not (baseline.event_id.equals(proposed.event_id)
            and baseline.event_id.equals(kbrd.event_id)):
        raise ValueError("Event IDs are not exactly aligned.")
    if len(baseline) != 4003 or baseline.conv_id.nunique() != 1283:
        raise ValueError("Expected 4,003 events and 1,283 conversations.")
    for frame in (baseline, proposed):
        if not np.array_equal(frame.conv_id.to_numpy(), kbrd.conv_id.to_numpy()):
            raise ValueError("Conversation mapping mismatch.")
        if not np.array_equal(frame.target_movie_id.to_numpy(),
                              kbrd.target_movie_id.to_numpy()):
            raise ValueError("Target mapping mismatch.")

    all_cells = set()
    for value in kbrd.top_movie_ids:
        for movie_id in inference.parse_ids(value):
            movie_id = int(movie_id)
            for category in categories.get(movie_id, []):
                all_cells.add((category, int(item2group[movie_id])))
    cells = sorted(all_cells)

    _, convs, bh, bn, be, bo = inference.build_event_contributions(
        kbrd, baseline, item2group, categories, cells, ref
    )
    _, p_convs, ph, pn, pe, po = inference.build_event_contributions(
        kbrd, proposed, item2group, categories, cells, ref
    )
    if not np.array_equal(convs, p_convs) or not np.allclose(bo, po):
        raise ValueError("Paired conversation/opportunity alignment failed.")

    unique_convs = np.unique(convs)
    conv_to_idx = {int(c): i for i, c in enumerate(unique_convs)}
    event_cluster = np.asarray([conv_to_idx[int(c)] for c in convs], dtype=int)
    n_clusters = len(unique_convs)
    n_events = len(convs)
    cluster_n = np.bincount(event_cluster, minlength=n_clusters).astype(float)

    c_bh = cluster_sum(bh[:, None], event_cluster, n_clusters)[:, 0]
    c_ph = cluster_sum(ph[:, None], event_cluster, n_clusters)[:, 0]
    c_bn = cluster_sum(bn[:, None], event_cluster, n_clusters)[:, 0]
    c_pn = cluster_sum(pn[:, None], event_cluster, n_clusters)[:, 0]
    c_be = cluster_sum(be, event_cluster, n_clusters)
    c_pe = cluster_sum(pe, event_cluster, n_clusters)
    c_opp = cluster_sum(bo, event_cluster, n_clusters)

    support = (c_opp > EPS).sum(axis=0)
    cell_mask = support >= args.minimum_cell_conversations
    if cell_mask.sum() < 2:
        raise RuntimeError("Fewer than two fairness cells pass support threshold.")

    total_opp = bo.sum(axis=0)
    observed = {
        "HR@10": float(ph.mean() - bh.mean()),
        "NDCG@10": float(pn.mean() - bn.mean()),
        "fairness_MAD": (
            inference.mad_from_totals(pe.sum(axis=0), total_opp, cell_mask)
            - inference.mad_from_totals(be.sum(axis=0), total_opp, cell_mask)
        ),
    }

    # Conversation cluster sizes.
    size_freq = pd.Series(cluster_n.astype(int)).value_counts().sort_index()
    cluster_sizes = pd.DataFrame({
        "conv_id": unique_convs,
        "events": cluster_n.astype(int),
    })
    args.out.mkdir(parents=True, exist_ok=True)
    cluster_sizes.to_csv(args.out / "conversation_cluster_sizes.csv", index=False)

    # Recreate the original cluster bootstrap from the exact frozen sufficient
    # statistics, using the original seed and algorithm.
    rng = np.random.default_rng(BOOT_SEED)
    boot = {
        "HR@10": np.empty(args.replicates),
        "NDCG@10": np.empty(args.replicates),
        "fairness_MAD": np.empty(args.replicates),
    }
    for start in range(0, args.replicates, 250):
        stop = min(start + 250, args.replicates)
        m = stop - start
        sampled = rng.integers(0, n_clusters, size=(m, n_clusters))
        mult = np.zeros((m, n_clusters), dtype=np.float64)
        rows = np.repeat(np.arange(m), n_clusters)
        np.add.at(mult, (rows, sampled.ravel()), 1.0)
        denom = mult @ cluster_n
        boot["HR@10"][start:stop] = (
            mult @ (c_ph - c_bh)
        ) / denom
        boot["NDCG@10"][start:stop] = (
            mult @ (c_pn - c_bn)
        ) / denom
        b_exp = mult @ c_be
        p_exp = mult @ c_pe
        b_opp = mult @ c_opp
        for j in range(m):
            boot["fairness_MAD"][start + j] = (
                inference.mad_from_totals(p_exp[j], b_opp[j], cell_mask)
                - inference.mad_from_totals(b_exp[j], b_opp[j], cell_mask)
            )

    # Verify regenerated draws against the archived arrays.
    with np.load(args.bootstrap_archive, allow_pickle=False) as archive:
        archive_keys = {
            "HR@10": "delta_hr10",
            "NDCG@10": "delta_ndcg10",
            "fairness_MAD": "delta_fairness_mad",
        }
        archive_comparison = {}
        for metric, key in archive_keys.items():
            saved = np.asarray(archive[key], dtype=float)
            regenerated = boot[metric]
            same_shape = saved.shape == regenerated.shape
            max_abs = (float(np.max(np.abs(saved - regenerated)))
                       if same_shape else None)
            archive_comparison[metric] = {
                "archive_shape": list(saved.shape),
                "regenerated_shape": list(regenerated.shape),
                "same_shape": bool(same_shape),
                "max_absolute_difference": max_abs,
                "exact_match": bool(same_shape and np.array_equal(saved, regenerated)),
                "allclose": bool(same_shape and np.allclose(
                    saved, regenerated, rtol=1e-10, atol=1e-12
                )),
            }

    # Delete-one-conversation jackknife for BCa intervals.
    jack = {k: np.empty(n_clusters) for k in observed}
    total_bh, total_ph = float(bh.sum()), float(ph.sum())
    total_bn, total_pn = float(bn.sum()), float(pn.sum())
    full_b_exp, full_p_exp = be.sum(axis=0), pe.sum(axis=0)
    for i in range(n_clusters):
        remaining_events = n_events - cluster_n[i]
        jack["HR@10"][i] = (
            (total_ph - c_ph[i]) - (total_bh - c_bh[i])
        ) / remaining_events
        jack["NDCG@10"][i] = (
            (total_pn - c_pn[i]) - (total_bn - c_bn[i])
        ) / remaining_events
        j_opp = total_opp - c_opp[i]
        j_b = full_b_exp - c_be[i]
        j_p = full_p_exp - c_pe[i]
        jack["fairness_MAD"][i] = (
            inference.mad_from_totals(j_p, j_opp, cell_mask)
            - inference.mad_from_totals(j_b, j_opp, cell_mask)
        )

    summaries = [
        distribution_summary(metric, boot[metric], observed[metric], jack[metric])
        for metric in ("HR@10", "NDCG@10", "fairness_MAD")
    ]
    pd.DataFrame([
        {k: v for k, v in row.items() if k != "bca"}
        | {
            "bca_available": row["bca"].get("available", False),
            "bca_ci95_low": row["bca"].get("low"),
            "bca_ci95_high": row["bca"].get("high"),
            "bca_z0": row["bca"].get("z0"),
            "bca_acceleration": row["bca"].get("acceleration"),
        }
        for row in summaries
    ]).to_csv(args.out / "bootstrap_distribution_and_ci_audit.csv", index=False)

    # Paired cluster sign-flip / label-swap diagnostic. It preserves pairing
    # within each conversation and recomputes the nonlinear fairness statistic.
    # It is NOT a confirmatory p-value: method assignment was not randomized,
    # and the sequential global ledger creates cross-conversation dependence.
    rng_perm = np.random.default_rng(PERM_SEED)
    n_perm = args.replicates
    perm_deltas = {k: np.empty(n_perm) for k in observed}
    mean_exp = 0.5 * (full_b_exp + full_p_exp)
    half_exp_diff = 0.5 * (c_pe - c_be)
    opp_total = total_opp

    for r in range(n_perm):
        signs = rng_perm.choice(np.array([-1.0, 1.0]), size=n_clusters)
        signed_h = float(np.dot(signs, c_ph - c_bh))
        signed_n = float(np.dot(signs, c_pn - c_bn))
        perm_deltas["HR@10"][r] = signed_h / n_events
        perm_deltas["NDCG@10"][r] = signed_n / n_events
        signed_exp = signs @ half_exp_diff
        pseudo_proposed = mean_exp + signed_exp
        pseudo_baseline = mean_exp - signed_exp
        perm_deltas["fairness_MAD"][r] = (
            inference.mad_from_totals(pseudo_proposed, opp_total, cell_mask)
            - inference.mad_from_totals(pseudo_baseline, opp_total, cell_mask)
        )

    perm_rows = []
    for metric in observed:
        vals = perm_deltas[metric]
        p = (1 + np.count_nonzero(np.abs(vals) >= abs(observed[metric]))) / (n_perm + 1)
        perm_rows.append({
            "metric": metric,
            "observed_delta": observed[metric],
            "permutation_replicates": n_perm,
            "two_sided_cluster_swap_p_exploratory": float(p),
            "valid_as_confirmatory_inference": False,
            "reason": (
                "No randomized method assignment; the proposed outcomes arise "
                "from a sequential global exposure ledger, so independent "
                "conversation-level label exchangeability is not established."
            ),
        })
    pd.DataFrame(perm_rows).to_csv(
        args.out / "paired_cluster_swap_sensitivity.csv", index=False
    )

    # Explicitly inspect the event-order/global-ledger structure from source.
    sequential_audit = inspect_sequential_design(
        args.script_22_24b, args.gate_script
    )

    # Summarize why bootstrap mean can differ: MAD is nonlinear in aggregate
    # exposure/opportunity ratios, so E[MAD(sample)] need not equal MAD(full data).
    fairness_row = next(x for x in summaries if x["metric"] == "fairness_MAD")
    bootstrap_bias_note = {
        "observed_fairness_delta": observed["fairness_MAD"],
        "bootstrap_mean": fairness_row["bootstrap_mean"],
        "bootstrap_mean_minus_observed": fairness_row[
            "bootstrap_bias_mean_minus_observed"
        ],
        "bootstrap_median": fairness_row["bootstrap_median"],
        "bootstrap_skewness": fairness_row["bootstrap_skewness"],
        "explanation": (
            "Fairness MAD is a nonlinear function of aggregate exposure divided "
            "by aggregate opportunity, followed by a mean absolute deviation "
            "across cells. The expectation of this nonlinear statistic under "
            "cluster resampling need not equal its value at the observed aggregate "
            "totals. The discrepancy is therefore a bootstrap bias/curvature "
            "diagnostic, not by itself evidence of a coding error. It must be "
            "interpreted alongside exact archive reproduction and the jackknife."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"events_per_conversation": int(k), "conversation_count": int(v)}
        for k, v in size_freq.items()
    ]).to_csv(args.out / "cluster_size_frequency.csv", index=False)

    summary = {
        "experiment": "22.24C_paired_cluster_inference_validation",
        "diagnostic_only": True,
        "events": n_events,
        "conversations": n_clusters,
        "fairness_cells_total": len(cells),
        "fairness_cells_included": int(cell_mask.sum()),
        "minimum_cell_conversation_support": args.minimum_cell_conversations,
        "cluster_size": {
            "mean": float(cluster_n.mean()),
            "median": float(np.median(cluster_n)),
            "sd": float(cluster_n.std(ddof=1)),
            "min": int(cluster_n.min()),
            "max": int(cluster_n.max()),
            "single_event_conversations": int(np.sum(cluster_n == 1)),
            "multi_event_conversations": int(np.sum(cluster_n > 1)),
            "frequency": {str(int(k)): int(v) for k, v in size_freq.items()},
        },
        "observed_deltas": observed,
        "bootstrap_archive_reproduction": archive_comparison,
        "bootstrap_distribution_summaries": summaries,
        "fairness_bootstrap_discrepancy": bootstrap_bias_note,
        "paired_cluster_swap_sensitivity": {
            "results_file": "paired_cluster_swap_sensitivity.csv",
            "confirmatory": False,
            "reason": (
                "The 22.24B comparison is not a randomized treatment assignment. "
                "In addition, the proposed ranking policy uses a global sequential "
                "ledger, so independent cluster exchangeability is not established."
            ),
        },
        "sequential_policy_audit": sequential_audit,
        "inference_conclusion": {
            "conversation_bootstrap": (
                "Conditional on frozen event rankings; its policy-level validity "
                "is not established because the sequential global ledger couples "
                "outcomes across conversations."
            ),
            "paired_cluster_permutation": (
                "Not defensible as confirmatory policy-level inference from these "
                "frozen outputs alone. The computed cluster-swap result is a "
                "sensitivity diagnostic only."
            ),
            "holm_adjustment": (
                "Not applied to exploratory permutation values. Multiplicity "
                "correction cannot repair invalid exchangeability assumptions."
            ),
            "policy_level_uncertainty": (
                "Cannot be recovered from frozen rankings alone without modeling "
                "or rerunning the sequential policy under alternative sequences; "
                "that is outside this diagnostic and is not performed."
            ),
        },
        "prohibited_actions_performed": {
            "training": False,
            "retrieval_rerun": False,
            "ranking_policy_rerun": False,
            "independent_test_rerun": False,
            "original_outputs_overwritten": False,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired_cluster_inference_validation.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(summaries).drop(columns=["bca"]).to_csv(
        args.out / "bootstrap_ci_full_summary.csv", index=False
    )

    print("\n========== CLUSTER STRUCTURE ==========")
    print(json.dumps(summary["cluster_size"], indent=2))
    print("\n========== OBSERVED DELTAS ==========")
    print(json.dumps(observed, indent=2))
    print("\n========== BOOTSTRAP ARCHIVE REPRODUCTION ==========")
    print(json.dumps(archive_comparison, indent=2))
    print("\n========== BOOTSTRAP DISTRIBUTION ==========")
    print(pd.DataFrame([
        {k: v for k, v in row.items() if k != "bca"}
        | {
            "bca_available": row["bca"].get("available", False),
            "bca_low": row["bca"].get("low"),
            "bca_high": row["bca"].get("high"),
        }
        for row in summaries
    ]).to_string(index=False))
    print("\n========== SEQUENTIAL LEDGER AUDIT ==========")
    print(json.dumps(sequential_audit, indent=2))
    print("\n========== INFERENCE CONCLUSION ==========")
    print(json.dumps(summary["inference_conclusion"], indent=2))
    print("\nOutput:", args.out.resolve())
    print("Diagnostic complete. Original rankings and outputs were not modified.")


if __name__ == "__main__":
    main()
