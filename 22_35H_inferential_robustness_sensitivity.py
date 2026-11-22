#!/usr/bin/env python
"""
Experiment 22.35H — Inferential Robustness and Sensitivity Analysis.

Consumes frozen 22.35F and 22.35G validation artifacts. It does not train,
retrieve, rerank, retune, modify checkpoints, or access the independent test.

Primary estimand: the proposal-defined cumulative category-conditioned fairness
loss from 22.35F, evaluated after warm-up + system-specific tuning + held-out
history. R0_filtered is primary; R1_raw is the prespecified KG-variant
sensitivity analysis for A3 using R0-selected parameters.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

SYSTEMS = ("A0_KGSF", "A1_22_16", "A2_22_18", "A3_KG_FairCRS")
COMPARATORS = SYSTEMS[:3]
METRICS = ("HR@10", "NDCG@10", "MRR@10", "L_fair")
EPS = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    path.write_text(json.dumps(safe(obj), indent=2, allow_nan=False), encoding="utf-8")


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def bca_interval(bootstrap, observed, jackknife, alpha=0.05):
    """BCa interval using paired cluster-bootstrap draws and delete-one-cluster jackknife."""
    b = np.asarray(bootstrap, dtype=float)
    j = np.asarray(jackknife, dtype=float)
    b, j = b[np.isfinite(b)], j[np.isfinite(j)]
    if len(b) < 1000 or len(j) < 20:
        return {"available": False, "reason": "Insufficient finite bootstrap or jackknife values"}
    prop = (np.count_nonzero(b < observed) + 0.5 * np.count_nonzero(b == observed)) / len(b)
    prop = float(np.clip(prop, 1.0 / (2 * len(b)), 1 - 1.0 / (2 * len(b))))
    z0 = NormalDist().inv_cdf(prop)
    jbar = float(j.mean())
    dif = jbar - j
    denom = 6.0 * (float(np.sum(dif ** 2)) ** 1.5)
    accel = float(np.sum(dif ** 3) / denom) if denom > 0 else 0.0
    qs = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = NormalDist().inv_cdf(q)
        den = 1 - accel * (z0 + zq)
        if abs(den) < 1e-12:
            return {"available": False, "reason": "Singular BCa transformation",
                    "bias_correction_z0": z0, "acceleration": accel}
        qs.append(float(np.clip(NormalDist().cdf(z0 + (z0 + zq) / den), 0, 1)))
    lo, hi = np.quantile(b, qs)
    return {"available": True, "low": float(lo), "high": float(hi),
            "bias_correction_z0": float(z0), "acceleration": float(accel),
            "adjusted_quantiles": qs}


def distribution_diagnostics(x, observed):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    centered = x - x.mean()
    var = float(np.mean(centered ** 2))
    return {
        "n": int(len(x)),
        "observed_delta": float(observed),
        "bootstrap_mean": float(x.mean()),
        "bootstrap_median": float(np.median(x)),
        "bootstrap_sd": float(np.std(x, ddof=1)),
        "bootstrap_skewness": float(np.mean(centered ** 3) / var ** 1.5) if var > 0 else 0.0,
        "fraction_below_zero": float(np.mean(x < 0)),
        "fraction_equal_zero": float(np.mean(x == 0)),
        "fraction_above_zero": float(np.mean(x > 0)),
        "q025": float(np.quantile(x, .025)),
        "q975": float(np.quantile(x, .975)),
        "q001": float(np.quantile(x, .001)),
        "q01": float(np.quantile(x, .01)),
        "q05": float(np.quantile(x, .05)),
        "q50": float(np.quantile(x, .50)),
        "q95": float(np.quantile(x, .95)),
        "q99": float(np.quantile(x, .99)),
        "q999": float(np.quantile(x, .999)),
    }


def cluster_sign_flip_test(cluster_event_sums, total_events, n_perm, rng):
    """
    Two-sided paired cluster sign-flip test for event-averaged utility.
    Each conversation contributes the sum of its paired event-level differences;
    the statistic divides the signed cluster totals by the total event count,
    preserving the proposal's event-level mean utility estimand.
    """
    x = np.asarray(cluster_event_sums, dtype=float)
    if total_events <= 0:
        raise ValueError("total_events must be positive")
    observed = float(x.sum() / total_events)
    exceed = 0
    done = 0
    batch = 5000
    while done < n_perm:
        n = min(batch, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(x)))
        stats = (signs * x[None, :]).sum(axis=1) / total_events
        exceed += int(np.sum(np.abs(stats) >= abs(observed) - 1e-15))
        done += n
    return observed, float((exceed + 1) / (n_perm + 1))


def main():
    root = Path(".")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f-script", type=Path, default=root / "22_35F_validated_fairness_evaluation.py")
    ap.add_argument("--ref16", type=Path, default=root / "22_16_category_context_utility_calibration.py")
    ap.add_argument("--ref18", type=Path, default=root / "22_18_contextual_relevance_gate_validation.py")
    ap.add_argument("--kgsf", type=Path, default=root / "results_corrected/experiment_22_35/kgsf_validation/kgsf_validation_top500.csv")
    ap.add_argument("--q-r0", type=Path, default=root / "results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R0_filtered.csv")
    ap.add_argument("--q-r1", type=Path, default=root / "results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R1_raw.csv")
    ap.add_argument("--catalog", type=Path, default=root / "catalog_audit_corrected.pkl")
    ap.add_argument("--profiles", type=Path, default=root / "results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
    ap.add_argument("--f-dir", type=Path, default=root / "results_corrected/experiment_22_35/validated_fairness_evaluation")
    ap.add_argument("--g-dir", type=Path, default=root / "results_corrected/experiment_22_35/direct_paired_statistical_comparison")
    ap.add_argument("--out", type=Path, default=root / "results_corrected/experiment_22_35/inferential_robustness_sensitivity")
    ap.add_argument("--signflip-reps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=22358)
    a = ap.parse_args()

    files = [a.f_script, a.ref16, a.ref18, a.kgsf, a.q_r0, a.q_r1, a.catalog, a.profiles,
             a.f_dir / "event_level_system_comparisons.csv",
             a.f_dir / "parameter_decisions.json",
             a.f_dir / "tuning_grid_results.csv",
             a.f_dir / "cumulative_fairness_cells.csv",
             a.f_dir / "split_and_cumulative_summaries.csv",
             a.g_dir / "paired_bootstrap_replicates.csv",
             a.g_dir / "paired_contrast_summary.csv",
             a.g_dir / "validation_checks.json"]
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(f"Required 22.35H input missing: {p}")
    if a.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing 22.35H output: {a.out}")
    if a.signflip_reps < 10000:
        raise ValueError("--signflip-reps must be at least 10,000.")

    # Verify 22.35G and 22.35F are complete before using their outputs.
    gchecks = json.loads((a.g_dir / "validation_checks.json").read_text(encoding="utf-8"))
    if not gchecks.get("overall_pass", False):
        raise RuntimeError("22.35G validation_checks.json does not pass.")
    fchecks = json.loads((a.f_dir / "validation_checks.json").read_text(encoding="utf-8"))
    if not fchecks.get("overall_pass", False):
        raise RuntimeError("22.35F validation_checks.json does not pass.")
    if gchecks.get("primary_variant") != "R0_filtered":
        raise RuntimeError("22.35G primary variant is not R0_filtered.")

    F = load_module(a.f_script, "frozen_2235f_for_2235h")
    ref = F.import_ref(a.ref16, "ref16_for_2235h", ("load_metadata", "split_conversations", "ExposureLedger", "rerank"))
    ref18 = F.import_ref(a.ref18, "ref18_for_2235h", ("contextual_gate_rerank",))
    F.ref18 = ref18

    item2group, catalog, categories = ref.load_metadata(a.catalog, a.profiles)
    events, cands = F.load_data(a.kgsf, a.q_r0, catalog)
    events_r1, cands_r1 = F.load_data(a.kgsf, a.q_r1, catalog)
    if events != events_r1 or any(not np.array_equal(cands[e]["ids"], cands_r1[e]["ids"]) for e in events):
        raise RuntimeError("R0/R1 event and candidate alignment is not identical.")

    conv_ids = [v["conv"] for v in events.values()]
    wc, tc, hc = ref.split_conversations(conv_ids)
    if wc & tc or wc & hc or tc & hc:
        raise RuntimeError("Conversation splits overlap.")
    warm = sorted(e for e, v in events.items() if v["conv"] in wc)
    tune = sorted(e for e, v in events.items() if v["conv"] in tc)
    held = sorted(e for e, v in events.items() if v["conv"] in hc)
    if len(held) != 624 or len(hc) != 176:
        raise RuntimeError("Unexpected held-out split; expected 624 events / 176 conversations.")

    decisions = json.loads((a.f_dir / "parameter_decisions.json").read_text(encoding="utf-8"))
    event_df = pd.read_csv(a.f_dir / "event_level_system_comparisons.csv")
    event_df = event_df[event_df.variant == "R0_filtered"].copy()
    tuning = pd.read_csv(a.f_dir / "tuning_grid_results.csv")
    cells = pd.read_csv(a.f_dir / "cumulative_fairness_cells.csv")
    split_summary = pd.read_csv(a.f_dir / "split_and_cumulative_summaries.csv")
    gboot = pd.read_csv(a.g_dir / "paired_bootstrap_replicates.csv")
    required_event = {"system","event_id","conv_id","split","hit10","ndcg10","mrr10","top10_movie_ids"}
    if required_event - set(event_df.columns):
        raise RuntimeError(f"22.35F event file missing: {sorted(required_event-set(event_df.columns))}")
    event_df["event_id"] = event_df.event_id.astype(int)
    event_df["conv_id"] = event_df.conv_id.astype(int)

    # Rebuild exact warm+tune initial state using frozen 22.35F selected parameters.
    preheld, held_frames = {}, {}
    for system in SYSTEMS:
        wl = F.warm_legacy(warm, cands, item2group, categories, ref)
        wp = F.warm_prop(warm, cands, item2group, categories, float(decisions["A3_KG_FairCRS"]["parameters"]["q_threshold"]))
        params = decisions[system]["parameters"]
        _, tl, tp, _, _ = F.run_segment(system, tune, events, cands, item2group, categories,
                                         ref, wl, wp, params,
                                         float(decisions["A3_KG_FairCRS"]["parameters"]["q_threshold"]),
                                         float(decisions["A3_KG_FairCRS"]["parameters"]["R_min"]))
        z = event_df[(event_df.system == system) & (event_df.split == "heldout")].copy()
        if z.event_id.duplicated().any() or set(z.event_id) != set(held):
            raise RuntimeError(f"{system}: held-out event slate mismatch.")
        z = z.set_index("event_id", drop=False)
        for e in held:
            if int(z.loc[e, "conv_id"]) != events[e]["conv"]:
                raise RuntimeError(f"{system}: conversation ID mismatch at event {e}.")
            ids = F.parse_list(z.loc[e, "top10_movie_ids"], "top10_movie_ids")
            if len(ids) != 10 or len(set(map(int, ids))) != 10:
                raise RuntimeError(f"{system}: invalid held-out Top-10 slate at event {e}.")
        preheld[system] = tp.clone()
        held_frames[system] = z

    # Read G bootstrap draws and check complete paired replicate structure.
    required_boot = {"replicate","comparator","metric","A3_minus_comparator"}
    if required_boot - set(gboot.columns):
        raise RuntimeError(f"22.35G bootstrap file missing: {sorted(required_boot-set(gboot.columns))}")
    gboot["replicate"] = gboot.replicate.astype(int)
    if gboot.duplicated(["replicate","comparator","metric"]).any():
        raise RuntimeError("Duplicate 22.35G bootstrap keys.")
    nreps = int(gboot.replicate.nunique())
    expected_keys = {(c, m) for c in COMPARATORS for m in METRICS}
    for r in range(1, nreps + 1):
        got = set(map(tuple, gboot[gboot.replicate == r][["comparator","metric"]].to_numpy()))
        if got != expected_keys:
            raise RuntimeError(f"Incomplete paired bootstrap replicate {r}.")
    boot = {(c, m): gboot[(gboot.comparator == c) & (gboot.metric == m)]
            .sort_values("replicate")["A3_minus_comparator"].to_numpy(float)
            for c in COMPARATORS for m in METRICS}

    # Compute observed estimates and delete-one-conversation jackknife estimates.
    byconv = defaultdict(list)
    for e in held:
        byconv[int(events[e]["conv"])].append(e)
    held_convs = sorted(byconv)
    observed, jack = {}, {}
    for system in SYSTEMS:
        z = held_frames[system]
        utility = np.asarray(z.loc[held, ["hit10","ndcg10","mrr10"]], dtype=float)
        state = preheld[system].clone()
        for e in held:
            state.add_R(cands[e], item2group, categories,
                        float(decisions["A3_KG_FairCRS"]["parameters"]["q_threshold"]))
            slate = list(map(int, F.parse_list(z.loc[e, "top10_movie_ids"], "top10_movie_ids")))
            state.add_E(slate, item2group, categories)
        fair, nactive, _ = F.proposal_terms(state, float(decisions["A3_KG_FairCRS"]["parameters"]["R_min"]))
        observed[system] = {"HR@10":float(utility[:,0].mean()),"NDCG@10":float(utility[:,1].mean()),
                            "MRR@10":float(utility[:,2].mean()),"L_fair":float(fair),
                            "active_categories":int(nactive)}
        jack[system] = {m:[] for m in METRICS}
        for omitted in held_convs:
            keep = [e for e in held if int(events[e]["conv"]) != omitted]
            vals = np.asarray(z.loc[keep, ["hit10","ndcg10","mrr10"]], dtype=float)
            for idx, m in enumerate(("HR@10","NDCG@10","MRR@10")):
                jack[system][m].append(float(vals[:,idx].mean()))
            st = preheld[system].clone()
            for e in keep:
                st.add_R(cands[e], item2group, categories,
                         float(decisions["A3_KG_FairCRS"]["parameters"]["q_threshold"]))
                slate = list(map(int, F.parse_list(z.loc[e, "top10_movie_ids"], "top10_movie_ids")))
                st.add_E(slate, item2group, categories)
            jf, jn, _ = F.proposal_terms(st, float(decisions["A3_KG_FairCRS"]["parameters"]["R_min"]))
            if not np.isfinite(jf) or jn < 1:
                raise RuntimeError(f"Jackknife produced unsupported fairness cells for {system}, omit {omitted}.")
            jack[system]["L_fair"].append(float(jf))

    # BCa intervals and bootstrap distribution diagnostics for all 12 paired contrasts.
    infer_rows, dist_rows = [], []
    for comp in COMPARATORS:
        for metric in METRICS:
            obs_delta = observed["A3_KG_FairCRS"][metric] - observed[comp][metric]
            jk_delta = np.asarray(jack["A3_KG_FairCRS"][metric]) - np.asarray(jack[comp][metric])
            b = boot[(comp, metric)]
            if len(b) != nreps or not np.isfinite(b).all():
                raise RuntimeError(f"Invalid bootstrap draws for {comp}/{metric}.")
            bca = bca_interval(b, obs_delta, jk_delta)
            pct = np.quantile(b, [.025,.975])
            basic = [2*obs_delta-pct[1], 2*obs_delta-pct[0]]
            infer_rows.append({"comparator":comp,"metric":metric,"observed_A3_minus_comparator":obs_delta,
                "bootstrap_reps":len(b),"percentile_ci95_low":float(pct[0]),"percentile_ci95_high":float(pct[1]),
                "basic_ci95_low":float(basic[0]),"basic_ci95_high":float(basic[1]),
                "bca_available":bca["available"],"bca_ci95_low":bca.get("low",np.nan),
                "bca_ci95_high":bca.get("high",np.nan),"bca_bias_correction_z0":bca.get("bias_correction_z0",np.nan),
                "bca_acceleration":bca.get("acceleration",np.nan),"jackknife_clusters":len(jk_delta)})
            dist_rows.append({"comparator":comp,"metric":metric,**distribution_diagnostics(b,obs_delta)})

    # Paired cluster sign-flip tests are restricted to recommendation utility.
    # Fairness inference remains based on the nonlinear cumulative bootstrap + BCa intervals.
    rng = np.random.default_rng(a.seed)
    perm_rows = []
    for comp in COMPARATORS:
        for metric in ("HR@10","NDCG@10","MRR@10"):
            idx = ("HR@10","NDCG@10","MRR@10").index(metric)
            cluster_diffs = []
            for conv in held_convs:
                es = byconv[conv]
                av = np.asarray([held_frames["A3_KG_FairCRS"].loc[e, ["hit10","ndcg10","mrr10"]].to_numpy(float)[idx] for e in es])
                bv = np.asarray([held_frames[comp].loc[e, ["hit10","ndcg10","mrr10"]].to_numpy(float)[idx] for e in es])
                cluster_diffs.append(float(np.sum(av - bv)))
            total_held_events = sum(len(byconv[c]) for c in held_convs)
            est, p = cluster_sign_flip_test(cluster_diffs, total_held_events, a.signflip_reps, rng)
            perm_rows.append({"comparator":comp,"metric":metric,"event_weighted_mean_difference":est,
                              "two_sided_cluster_signflip_p":p,"permutations":a.signflip_reps,
                              "clusters":len(held_convs),"events":total_held_events,
                              "assumption":"paired conversation-level sign exchangeability under null"})
    perm = pd.DataFrame(perm_rows)
    perm["holm_adjusted_p_9_utility_tests"] = holm(perm.two_sided_cluster_signflip_p.to_numpy())

    # Tuning-grid sensitivity: validation tuning conversations only; no retuning occurs here.
    a3grid = tuning[tuning.system == "A3_KG_FairCRS"].copy()
    needed_grid = {"q_threshold","R_min","lam","L_fair_cumulative","NDCG_retention","utility_constraint_met","active_categories_cumulative"}
    if needed_grid - set(a3grid.columns):
        raise RuntimeError(f"Tuning grid missing columns: {sorted(needed_grid-set(a3grid.columns))}")
    feasible = a3grid[(a3grid.utility_constraint_met.astype(bool)) &
                      (a3grid.active_categories_cumulative.astype(int) > 0) &
                      np.isfinite(pd.to_numeric(a3grid.L_fair_cumulative,errors="coerce"))].copy()
    if feasible.empty:
        raise RuntimeError("No feasible A3 tuning-grid rows; cannot report parameter robustness.")
    best_loss = float(feasible.L_fair_cumulative.min())
    feasible["loss_regret_from_best"] = feasible.L_fair_cumulative - best_loss
    feasible["within_1pct_of_best_loss"] = feasible.loss_regret_from_best <= max(abs(best_loss)*.01, 1e-12)
    feasible["within_5pct_of_best_loss"] = feasible.loss_regret_from_best <= max(abs(best_loss)*.05, 1e-12)
    param_rows = []
    for param in ("q_threshold","R_min","lam"):
        for value, sub in feasible.groupby(param):
            param_rows.append({"parameter":param,"value":float(value),"feasible_grid_rows":len(sub),
                "minimum_fairness_loss":float(sub.L_fair_cumulative.min()),
                "median_fairness_loss":float(sub.L_fair_cumulative.median()),
                "maximum_fairness_loss":float(sub.L_fair_cumulative.max()),
                "share_within_1pct_of_best":float(sub.within_1pct_of_best_loss.mean()),
                "share_within_5pct_of_best":float(sub.within_5pct_of_best_loss.mean()),
                "mean_ndcg_retention":float(sub.NDCG_retention.mean())})
    param_sensitivity = pd.DataFrame(param_rows)
    selected = decisions["A3_KG_FairCRS"]["parameters"]
    selected_match = feasible[(np.isclose(feasible.q_threshold.astype(float),float(selected["q_threshold"]))) &
        (np.isclose(feasible.R_min.astype(float),float(selected["R_min"]))) &
        (np.isclose(feasible.lam.astype(float),float(selected["lam"])))]
    if selected_match.empty:
        raise RuntimeError("Selected A3 parameters are not present among feasible tuning-grid rows.")

    # R1 is sensitivity only: reuse the R0-selected parameters; compare frozen F summaries.
    a3s = split_summary[(split_summary.system=="A3_KG_FairCRS") &
                        (split_summary.split=="heldout")].copy()
    if set(a3s.variant) != {"R0_filtered","R1_raw"}:
        raise RuntimeError("Expected A3 held-out R0 and R1 summaries in 22.35F.")
    r1_sensitivity = a3s[["variant","HR@10","NDCG@10","MRR@10","L_fair_cumulative",
                           "active_categories_cumulative","mean_kg_supported_top10",
                           "mean_eligible_candidates","mean_fallback_positions"]].copy()

    # Category-level decomposition; category contributions are identical across group rows,
    # so retain one contribution per category/system/scope to avoid double counting.
    scope_col = "cumulative_scope"
    target_scope = "warm_plus_tune_plus_heldout"
    cat = cells[cells[scope_col] == target_scope].copy()
    if not {"system","variant","category","category_variance_contribution"}.issubset(cat.columns):
        raise RuntimeError("Cumulative fairness cell export lacks proposal category contributions.")
    cat = cat.drop_duplicates(["system","variant","category"])
    cat_r0 = cat[cat.variant=="R0_filtered"].pivot(index="category",columns="system",
                                                   values="category_variance_contribution")
    cat_rows=[]
    for comp in COMPARATORS:
        if comp not in cat_r0 or "A3_KG_FairCRS" not in cat_r0:
            raise RuntimeError("Missing system in category contribution table.")
        z=cat_r0[["A3_KG_FairCRS",comp]].dropna()
        for category,row in z.iterrows():
            cat_rows.append({"category":category,"comparator":comp,
                "A3_category_variance_contribution":float(row["A3_KG_FairCRS"]),
                "comparator_category_variance_contribution":float(row[comp]),
                "A3_minus_comparator":float(row["A3_KG_FairCRS"]-row[comp]),
                "direction":"improved" if row["A3_KG_FairCRS"]<row[comp] else
                            ("worsened" if row["A3_KG_FairCRS"]>row[comp] else "unchanged")})
    category_decomp = pd.DataFrame(cat_rows)

    # Sensitivity output retains the entire tuning grid; no new parameter selection.
    a.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(infer_rows).to_csv(a.out/"paired_bca_inference.csv",index=False)
    pd.DataFrame(dist_rows).to_csv(a.out/"bootstrap_distribution_diagnostics.csv",index=False)
    perm.to_csv(a.out/"paired_cluster_signflip_utility.csv",index=False)
    param_sensitivity.to_csv(a.out/"tuning_parameter_sensitivity.csv",index=False)
    feasible.to_csv(a.out/"feasible_a3_tuning_grid_with_regret.csv",index=False)
    r1_sensitivity.to_csv(a.out/"R0_R1_A3_sensitivity.csv",index=False)
    category_decomp.to_csv(a.out/"category_level_fairness_decomposition.csv",index=False)
    tuning.to_csv(a.out/"original_tuning_grid_preserved.csv",index=False)

    checks = {
        "experiment":"22.35H_inferential_robustness_sensitivity",
        "uses_validated_22_35F_and_22_35G_outputs":True,
        "independent_test_read":False,"training_performed":False,
        "retrieval_performed":False,"reranking_performed":False,
        "parameter_retuning_performed":False,
        "22_35E_outputs_modified":False,"22_35F_outputs_modified":False,"22_35G_outputs_modified":False,
        "R0_primary_R1_sensitivity_only":True,
        "R0_R1_candidate_alignment_verified":True,
        "warm_tune_histories_reconstructed_from_22_35F":True,
        "jackknife_unit":"conversation",
        "bca_intervals_use_22_35G_paired_cluster_bootstrap":True,
        "utility_signflip_is_conversation_clustered":True,
        "fairness_not_subjected_to_naive_event_level_permutation":True,
        "selected_parameters_found_in_feasible_tuning_grid":not selected_match.empty,
        "all_three_sensitivity_parameters_reported":set(param_sensitivity.parameter)=={"q_threshold","R_min","lam"},
        "R1_parameters_reused_without_retuning":True,
        "category_decomposition_uses_proposal_category_contributions":True,
        "bca_intervals_available":all(x["bca_available"] for x in infer_rows),
        "all_primary_comparisons_present":len(infer_rows)==12,
        "utility_signflip_tests_present":len(perm)==9,
        "category_comparisons_present":len(category_decomp)>0,
    }
    
    # Validate positive requirements, prohibited actions, and typed metadata
    # separately. "jackknife_unit" is a string-valued field, not a Boolean.
    required_true = {
        "uses_validated_22_35F_and_22_35G_outputs",
        "R0_primary_R1_sensitivity_only",
        "R0_R1_candidate_alignment_verified",
        "warm_tune_histories_reconstructed_from_22_35F",
        "bca_intervals_use_22_35G_paired_cluster_bootstrap",
        "utility_signflip_is_conversation_clustered",
        "fairness_not_subjected_to_naive_event_level_permutation",
        "selected_parameters_found_in_feasible_tuning_grid",
        "all_three_sensitivity_parameters_reported",
        "R1_parameters_reused_without_retuning",
        "category_decomposition_uses_proposal_category_contributions",
        "bca_intervals_available",
        "all_primary_comparisons_present",
        "utility_signflip_tests_present",
        "category_comparisons_present",
    }

    required_false = {
        "independent_test_read",
        "training_performed",
        "retrieval_performed",
        "reranking_performed",
        "parameter_retuning_performed",
        "22_35E_outputs_modified",
        "22_35F_outputs_modified",
        "22_35G_outputs_modified",
    }

    required_metadata = {"jackknife_unit"}
    expected = required_true | required_false | required_metadata
    missing = expected - set(checks)

    failed_true = [
        key for key in sorted(required_true)
        if checks.get(key) is not True
    ]

    failed_false = [
        key for key in sorted(required_false)
        if checks.get(key) is not False
    ]

    # This metadata must be the literal string "conversation".
    jackknife_unit_valid = checks.get("jackknife_unit") == "conversation"
    checks["jackknife_unit_valid"] = jackknife_unit_valid

    checks["missing_validation_conditions"] = sorted(missing)
    checks["failed_required_true"] = failed_true
    checks["failed_required_false"] = failed_false

    checks["overall_pass"] = (
        not missing
        and not failed_true
        and not failed_false
        and jackknife_unit_valid
    )

    write_json(a.out/"validation_checks.json",checks)
    input_hashes={str(p):sha256(p) for p in files}
    manifest={
        "experiment":"22.35H_inferential_robustness_sensitivity",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
        "scope":"frozen_validation_artifacts_only",
        "primary_fairness_estimand":"22.35F proposal-defined cumulative L_fair; warm-up + system-specific tune + held-out",
        "proposal_objective":"Uniform average across active categories of relevance-weighted within-category variance of rho_cg=E_cg/(R_cg+epsilon); active category requires at least two supported popularity groups and R_cg >= R_min.",
        "inference":{
            "fairness":"paired conversation-cluster bootstrap percentile/basic and BCa intervals; BCa acceleration from delete-one-heldout-conversation jackknife, preserving system-specific warm+tune initial histories",
            "utility":"paired conversation-cluster sign-flip test on cluster sums, normalized by total held-out events; exchangeability/symmetry assumption stated",
            "multiplicity":"Holm across 9 utility sign-flip tests; BCa intervals reported for 3 fairness and 9 utility contrasts",
            "not_claimed":"No exact permutation p-value is reported for nonlinear cumulative fairness."
        },
        "sensitivity":{
            "parameters":["q_threshold","R_min","lambda"],
            "source":"22.35F tuning grid; tuning conversations only; no new selection",
            "kg_variant":"R1_raw reported as sensitivity using R0-selected parameters without retuning",
            "category":"per-category proposal variance contribution compared across systems"
        },
        "bootstrap_reps_from_22_35G":int(nreps),"signflip_reps":a.signflip_reps,"seed":a.seed,
        "selected_A3_parameters":selected,"input_sha256":input_hashes,
        "validation_checks":checks,
        "outputs":["paired_bca_inference.csv","bootstrap_distribution_diagnostics.csv",
            "paired_cluster_signflip_utility.csv","tuning_parameter_sensitivity.csv",
            "feasible_a3_tuning_grid_with_regret.csv","R0_R1_A3_sensitivity.csv",
            "category_level_fairness_decomposition.csv","original_tuning_grid_preserved.csv",
            "validation_checks.json","experiment_manifest.json"]
    }
    write_json(a.out/"experiment_manifest.json",manifest)
    print("\n========== EXPERIMENT 22.35H COMPLETE ==========")
    print("\nPaired BCa inference:")
    print(pd.DataFrame(infer_rows).to_string(index=False))
    print("\nUtility cluster sign-flip tests:")
    print(perm.to_string(index=False))
    print("\nParameter sensitivity:")
    print(param_sensitivity.to_string(index=False))
    print("\nR0/R1 A3 sensitivity:")
    print(r1_sensitivity.to_string(index=False))
    print("\nValidation overall_pass:",checks["overall_pass"])
    print("Independent test: NOT READ | 22.35E/F/G outputs untouched")
    if not checks["overall_pass"]:
        raise RuntimeError("22.35H validation failed; inspect validation_checks.json")


if __name__ == "__main__":
    main()