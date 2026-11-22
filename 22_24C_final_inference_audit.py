#!/usr/bin/env python
"""
Experiment 22.24C — Final inference audit (diagnostic only).

Reads frozen bootstrap distributions and existing 22.24C outputs.
Does NOT retrain, rerank, regenerate bootstrap samples, or overwrite 22.24C.

Important:
- A bootstrap distribution of an estimator is primarily an uncertainty
  distribution. It is not automatically a null distribution.
- This script therefore does NOT manufacture hypothesis-test p-values from
  the bootstrap distribution.
- BCa intervals require a delete-one-independent-cluster jackknife. The
  required jackknife estimates are not stored in bootstrap_distributions.npz.
  BCa is reported as unavailable unless a compatible jackknife CSV is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_KEYS = {
    "HR@10": "delta_hr10",
    "NDCG@10": "delta_ndcg10",
    "fairness_MAD": "delta_fairness_mad",
}


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def basic_ci(observed, q025, q975):
    # Basic/reflected interval: [2*theta_hat - q975, 2*theta_hat - q025]
    return 2.0 * observed - q975, 2.0 * observed - q025


def bca_interval(values, observed, jackknife_values, alpha=0.05):
    """
    BCa interval using empirical bias correction and delete-one-cluster
    jackknife acceleration. Returns None if inputs are degenerate/invalid.
    """
    values = np.asarray(values, dtype=float)
    jack = np.asarray(jackknife_values, dtype=float)
    values = values[np.isfinite(values)]
    jack = jack[np.isfinite(jack)]

    if len(values) < 1000 or len(jack) < 3:
        return None
    if np.std(jack, ddof=1) == 0:
        return None

    # Mid-rank treatment of ties gives a stable empirical z0.
    prop_less = (np.sum(values < observed) + 0.5 * np.sum(values == observed)) / len(values)
    eps = 0.5 / len(values)
    prop_less = float(np.clip(prop_less, eps, 1.0 - eps))
    z0 = float(_norm_ppf(prop_less))

    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    denom = 6.0 * (float(np.sum(diffs ** 2)) ** 1.5)
    if denom <= 0:
        return None
    acceleration = float(np.sum(diffs ** 3) / denom)

    z_alpha = [_norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)]
    adjusted = []
    for z in z_alpha:
        den = 1.0 - acceleration * (z0 + z)
        if abs(den) < 1e-12:
            return None
        adjusted.append(_norm_cdf(z0 + (z0 + z) / den))

    lo_q, hi_q = [float(np.clip(q, 0.0, 1.0)) for q in adjusted]
    if lo_q >= hi_q:
        return None
    return {
        "low": float(np.quantile(values, lo_q)),
        "high": float(np.quantile(values, hi_q)),
        "z0": z0,
        "acceleration": acceleration,
        "adjusted_quantile_low": lo_q,
        "adjusted_quantile_high": hi_q,
        "jackknife_clusters": int(len(jack)),
    }


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    # Acklam approximation; avoids scipy dependency.
    p = float(p)
    if not 0 < p < 1:
        raise ValueError("Normal quantile input must be in (0, 1).")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def distribution_audit(name, values, observed, jackknife=None):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 1000:
        raise ValueError(f"{name}: expected >=1,000 finite bootstrap draws; got {len(x)}")

    q025, q975 = np.quantile(x, [0.025, 0.975])
    pctl_low, pctl_high = float(q025), float(q975)
    basic_low, basic_high = basic_ci(observed, pctl_low, pctl_high)
    bca = bca_interval(x, observed, jackknife) if jackknife is not None else None

    # Bootstrap estimation-error distribution, not a null distribution.
    errors = x - observed
    row = {
        "metric": name,
        "n_bootstrap": int(len(x)),
        "observed_delta": float(observed),
        "bootstrap_mean": float(np.mean(x)),
        "bootstrap_median": float(np.median(x)),
        "bootstrap_bias_mean_minus_observed": float(np.mean(x) - observed),
        "bootstrap_bias_median_minus_observed": float(np.median(x) - observed),
        "bootstrap_sd": float(np.std(x, ddof=1)),
        "skewness": float(pd.Series(x).skew()),
        "q001": float(np.quantile(x, 0.001)),
        "q01": float(np.quantile(x, 0.01)),
        "q025": pctl_low,
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "q975": pctl_high,
        "q99": float(np.quantile(x, 0.99)),
        "q999": float(np.quantile(x, 0.999)),
        "fraction_below_zero": float(np.mean(x < 0)),
        "fraction_above_zero": float(np.mean(x > 0)),
        "fraction_bootstrap_below_observed": float(np.mean(x < observed)),
        "fraction_error_abs_ge_observed_abs": float(
            np.mean(np.abs(errors) >= abs(observed))
        ),
        "percentile_ci95_low": pctl_low,
        "percentile_ci95_high": pctl_high,
        "basic_ci95_low": float(basic_low),
        "basic_ci95_high": float(basic_high),
        "bca_available": bool(bca is not None),
        "bca_ci95_low": bca["low"] if bca else None,
        "bca_ci95_high": bca["high"] if bca else None,
        "bca_z0": bca["z0"] if bca else None,
        "bca_acceleration": bca["acceleration"] if bca else None,
        "bca_jackknife_clusters": bca["jackknife_clusters"] if bca else None,
    }
    return row


def cluster_diagnostics(event_csv: Path | None):
    if event_csv is None:
        return None, "Not supplied: cluster sizes cannot be reconstructed."
    if not event_csv.exists():
        return None, f"Not found: {event_csv}"
    df = pd.read_csv(event_csv)
    if "conv_id" not in df.columns:
        return None, "CSV lacks conv_id; cluster sizes unavailable."
    sizes = df.groupby("conv_id", dropna=False).size().rename("events_per_conversation")
    desc = sizes.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    out = pd.DataFrame({
        "conv_id": sizes.index.astype(str),
        "events_per_conversation": sizes.values,
    })
    out["cluster_size_rank"] = out["events_per_conversation"].rank(method="average")
    return out, desc.to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment-dir", type=Path,
        default=Path("results_final_test_m2/experiment_22_24C"),
        help="Existing frozen 22.24C output directory."
    )
    ap.add_argument(
        "--bootstrap-npz", type=Path, default=None,
        help="Defaults to EXPERIMENT_DIR/bootstrap_distributions.npz."
    )
    ap.add_argument(
        "--assessment-json", type=Path, default=None,
        help="Defaults to EXPERIMENT_DIR/final_research_assessment.json."
    )
    ap.add_argument(
        "--baseline-events", type=Path, default=None,
        help="Optional baseline_event_results.csv for cluster-size diagnostics."
    )
    ap.add_argument(
        "--proposed-events", type=Path, default=None,
        help="Optional proposed_event_results.csv; used to verify same cluster sizes."
    )
    ap.add_argument(
        "--jackknife-csv", type=Path, default=None,
        help=("Optional CSV with one row per deleted conversation and columns "
              "metric, estimate. Required for BCa acceleration.")
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Defaults to EXPERIMENT_DIR/final_inference_audit."
    )
    args = ap.parse_args()

    exp_dir = args.experiment_dir
    npz_path = args.bootstrap_npz or exp_dir / "bootstrap_distributions.npz"
    assessment_path = args.assessment_json or exp_dir / "final_research_assessment.json"
    out_dir = args.out_dir or exp_dir / "final_inference_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.is_file():
        raise FileNotFoundError(f"Bootstrap archive not found: {npz_path}")

    assessment = load_json(assessment_path)
    observed_lookup = {
        "HR@10": assessment.get("statistical_results", [{}])[0].get(
            "delta_proposed_minus_baseline"
        ),
        "NDCG@10": assessment.get("statistical_results", [{}])[1].get(
            "delta_proposed_minus_baseline"
        ),
        "fairness_MAD": assessment.get("statistical_results", [{}])[2].get(
            "delta_proposed_minus_baseline"
        ),
    }

    # Fall back to explicit assessment-level fields if rows are unavailable.
    if observed_lookup["fairness_MAD"] is None:
        observed_lookup["fairness_MAD"] = assessment.get("fairness_delta")

    jackknife_map = {}
    if args.jackknife_csv and args.jackknife_csv.exists():
        jdf = pd.read_csv(args.jackknife_csv)
        required = {"metric", "estimate"}
        if not required.issubset(jdf.columns):
            raise ValueError("Jackknife CSV must contain columns: metric, estimate")
        for metric, part in jdf.groupby("metric"):
            jackknife_map[str(metric)] = part["estimate"].to_numpy(dtype=float)

    with np.load(npz_path, allow_pickle=False) as archive:
        available = set(archive.files)
        rows = []
        arrays = {}
        for metric, key in METRIC_KEYS.items():
            if key not in available:
                raise KeyError(f"{key} missing from {npz_path}; found {sorted(available)}")
            values = np.asarray(archive[key], dtype=float)
            arrays[metric] = values
            observed = safe_float(observed_lookup.get(metric))
            if observed is None:
                raise ValueError(
                    f"Observed delta for {metric} missing from {assessment_path}. "
                    "Provide the original final_research_assessment.json."
                )
            rows.append(distribution_audit(
                metric, values, observed, jackknife_map.get(metric)
            ))

    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "bootstrap_distribution_audit.csv", index=False)

    # CI comparison: all interval estimates are descriptive unless bootstrap
    # regularity assumptions are justified for the estimator.
    ci_cols = [
        "metric", "observed_delta", "percentile_ci95_low", "percentile_ci95_high",
        "basic_ci95_low", "basic_ci95_high", "bca_available",
        "bca_ci95_low", "bca_ci95_high",
    ]
    dist_df[ci_cols].to_csv(out_dir / "confidence_interval_comparison.csv", index=False)

    # Cluster size audit. Prefer baseline event manifest; verify proposed too.
    cluster_df, cluster_summary = cluster_diagnostics(args.baseline_events)
    cluster_note = cluster_summary
    if cluster_df is not None:
        cluster_df.to_csv(out_dir / "cluster_size_diagnostics.csv", index=False)
        if args.proposed_events and args.proposed_events.exists():
            p_df, p_summary = cluster_diagnostics(args.proposed_events)
            if p_df is not None:
                b_counts = cluster_df.set_index("conv_id")["events_per_conversation"].sort_index()
                p_counts = p_df.set_index("conv_id")["events_per_conversation"].sort_index()
                cluster_note = {
                    "baseline": cluster_summary,
                    "proposed": p_summary,
                    "same_conversation_ids_and_cluster_sizes": bool(
                        b_counts.equals(p_counts)
                    ),
                }
    else:
        # Always emit a machine-readable note even when no source CSV is provided.
        pd.DataFrame([{"status": str(cluster_summary)}]).to_csv(
            out_dir / "cluster_size_diagnostics.csv", index=False
        )

    # Do not claim inferential p-values. A centered bootstrap tail calculation
    # is not a calibrated null test for this nonlinear statistic.
    tests = []
    for _, r in dist_df.iterrows():
        tests.append({
            "metric": r["metric"],
            "observed_delta": r["observed_delta"],
            "null_test_status": "NOT_COMPUTED",
            "p_value": None,
            "reason": (
                "The saved bootstrap draws approximate the estimator's sampling "
                "distribution, not a null distribution. A valid paired cluster "
                "permutation/randomization test or a justified null-imposed "
                "bootstrap must be constructed from cluster-level sufficient "
                "contributions and the exact statistic."
            ),
            "directional_bootstrap_fraction_below_zero": r["fraction_below_zero"],
            "interpretation": (
                "Descriptive bootstrap sign fraction only; not a p-value."
            ),
        })
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out_dir / "null_hypothesis_tests.csv", index=False)

    # Holm is only validly applied to valid p-values. Since no valid null p-values
    # are computed here, do not publish adjusted p-values.
    holm_df = pd.DataFrame([{
        "metric": metric,
        "valid_p_value_available": False,
        "raw_p_value": None,
        "holm_adjusted_p": None,
        "significant_after_holm": None,
        "status": "NOT_COMPUTED — no defensible null-test p-value supplied",
    } for metric in METRIC_KEYS])
    holm_df.to_csv(out_dir / "holm_adjusted_results.csv", index=False)

    fairness_row = dist_df.loc[dist_df.metric == "fairness_MAD"].iloc[0]
    observed_fairness = float(fairness_row.observed_delta)
    fairness_reduction = assessment.get("fairness_reduction_percent")

    audit = {
        "audit": "22.24C_final_inference_audit",
        "inputs": {
            "bootstrap_npz": str(npz_path),
            "assessment_json": str(assessment_path),
            "baseline_events_csv": str(args.baseline_events) if args.baseline_events else None,
            "proposed_events_csv": str(args.proposed_events) if args.proposed_events else None,
            "jackknife_csv": str(args.jackknife_csv) if args.jackknife_csv else None,
        },
        "frozen_experiment_preserved": True,
        "training_performed": False,
        "retrieval_rerun": False,
        "ranking_policy_modified": False,
        "bootstrap_replicates": int(len(arrays["fairness_MAD"])),
        "fairness_observed_delta": observed_fairness,
        "fairness_reduction_percent_from_original_assessment": fairness_reduction,
        "fairness_bootstrap_mean": float(fairness_row.bootstrap_mean),
        "fairness_bootstrap_bias": float(
            fairness_row.bootstrap_bias_mean_minus_observed
        ),
        "fairness_bootstrap_sd": float(fairness_row.bootstrap_sd),
        "fairness_percentile_ci95": [
            float(fairness_row.percentile_ci95_low),
            float(fairness_row.percentile_ci95_high),
        ],
        "fairness_basic_ci95": [
            float(fairness_row.basic_ci95_low),
            float(fairness_row.basic_ci95_high),
        ],
        "fairness_bca_ci95": (
            [
                float(fairness_row.bca_ci95_low),
                float(fairness_row.bca_ci95_high),
            ] if bool(fairness_row.bca_available) else None
        ),
        "cluster_size_diagnostics": cluster_note,
        "null_tests": "Not computed: saved bootstrap distribution is not a null distribution.",
        "holm": "Not computed: Holm requires valid raw p-values from defensible null tests.",
        "interpretation": (
            "The bootstrap mean's displacement from the observed fairness effect "
            "is a diagnostic warning about bias/nonlinearity or estimand mismatch. "
            "It does not by itself invalidate the observed point estimate, nor does "
            "a percentile/basic interval or bootstrap sign fraction establish a "
            "null-hypothesis significance result. BCa requires delete-one-cluster "
            "jackknife estimates of the same fairness estimator."
        ),
        "next_required_for_final_significance": [
            "Reconstruct cluster-level sufficient contributions from the frozen event files.",
            "Compute a delete-one-conversation jackknife of the exact fairness delta.",
            "Specify a valid paired null procedure (e.g. conversation-level paired randomization/permutation, if exchangeability is justified).",
            "Recompute raw p-values and apply Holm across the prespecified family.",
        ],
    }
    with (out_dir / "final_inference_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, allow_nan=False)

    print("=" * 78)
    print("EXPERIMENT 22.24C — FINAL INFERENCE AUDIT")
    print("=" * 78)
    print(f"Bootstrap archive: {npz_path}")
    print(f"Output directory:  {out_dir.resolve()}")
    print("\nBOOTSTRAP DISTRIBUTION SUMMARY")
    display_cols = [
        "metric", "observed_delta", "bootstrap_mean",
        "bootstrap_bias_mean_minus_observed", "bootstrap_sd",
        "percentile_ci95_low", "percentile_ci95_high",
        "basic_ci95_low", "basic_ci95_high", "bca_available",
    ]
    print(dist_df[display_cols].to_string(index=False))
    print("\nINFERENCE SAFEGUARDS")
    print("- Null-hypothesis p-values: NOT COMPUTED from ordinary bootstrap draws.")
    print("- Holm-adjusted p-values: NOT COMPUTED without valid raw p-values.")
    print("- BCa: available only if a delete-one-conversation jackknife CSV is supplied.")
    print("- Original 22.24C outputs and frozen rankings were not modified.")
    print("\nCreated:")
    for name in [
        "bootstrap_distribution_audit.csv",
        "cluster_size_diagnostics.csv",
        "confidence_interval_comparison.csv",
        "null_hypothesis_tests.csv",
        "holm_adjusted_results.csv",
        "final_inference_audit.json",
    ]:
        print("  ", out_dir / name)


if __name__ == "__main__":
    main()
