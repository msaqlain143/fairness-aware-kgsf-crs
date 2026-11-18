#!/usr/bin/env python
"""Diagnostic-only reconstruction of Experiment 22.24C event and fairness totals."""
from __future__ import annotations
import argparse, importlib.util, json, math, ast
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12
N_EVENTS, N_CONV, MIN_SUPPORT = 4003, 1283, 20
D_RESULTS = Path("results_final_test_m2/experiment_22_24B")
D_KBRD = Path("results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv")
D_CATALOG = Path("catalog_audit_corrected.pkl")
D_PROFILES = Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
D_REF = Path("22_16_category_context_utility_calibration.py")
D_INF = Path("22_24C_independent_test_statistical_inference.py")
D_OUT = Path("results_final_test_m2/experiment_22_24C/event_reconstruction_audit")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def ids(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    try:
        x = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        x = json.loads(str(value))
    if not isinstance(x, (list, tuple)):
        raise ValueError("Expected a list in an ID column")
    return [int(v) for v in x]

def mad(exposure, opportunity, mask=None):
    valid = np.asarray(opportunity) > EPS
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if valid.sum() < 2:
        raise RuntimeError("Fewer than two supported cells")
    z = np.log((np.asarray(exposure)[valid] + EPS) /
               (np.asarray(opportunity)[valid] + EPS))
    return float(np.mean(np.abs(z - z.mean())))

def close(a, b):
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=1e-9))

def json_safe(value):
    """Convert NumPy values and non-finite floats to strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=D_RESULTS)
    ap.add_argument("--kbrd", type=Path, default=D_KBRD)
    ap.add_argument("--catalog-pickle", type=Path, default=D_CATALOG)
    ap.add_argument("--category-profiles", type=Path, default=D_PROFILES)
    ap.add_argument("--reference", type=Path, default=D_REF)
    ap.add_argument("--inference-script", type=Path, default=D_INF)
    ap.add_argument("--out", type=Path, default=D_OUT)
    a = ap.parse_args()

    bp, pp, apath = (a.results_dir / "baseline_event_results.csv",
                     a.results_dir / "proposed_event_results.csv",
                     a.results_dir / "aggregate_metrics.csv")
    for p in (bp, pp, apath, a.kbrd, a.catalog_pickle, a.category_profiles,
              a.reference, a.inference_script):
        if not p.is_file():
            raise FileNotFoundError(p)

    print("=" * 78)
    print("EXPERIMENT 22.24C — EVENT-LEVEL RECONSTRUCTION AUDIT")
    print("=" * 78)

    ref = load_module(a.reference, "audit_ref_2216")
    inf = load_module(a.inference_script, "audit_inf_2224c")
    item2group, catalog, categories = ref.load_metadata(a.catalog_pickle, a.category_profiles)

    b, p, k = pd.read_csv(bp), pd.read_csv(pp), pd.read_csv(a.kbrd)
    agg = pd.read_csv(apath)
    result_cols = {"event_id","conv_id","target_movie_id","hit10","ndcg10",
                   "top10_movie_ids","top10_group_ids"}
    k_cols = {"event_id","conv_id","target_movie_id","top_movie_ids"}
    for label, df in (("baseline",b),("proposed",p)):
        miss = result_cols - set(df.columns)
        if miss: raise ValueError(f"{label} missing {sorted(miss)}")
        if df.event_id.duplicated().any(): raise ValueError(f"Duplicate {label} event IDs")
    miss = k_cols - set(k.columns)
    if miss: raise ValueError(f"KBRD missing {sorted(miss)}")
    if k.event_id.duplicated().any(): raise ValueError("Duplicate KBRD event IDs")

    for df in (b,p,k):
        for c in ("event_id","conv_id","target_movie_id"):
            df[c] = df[c].astype(int)
        df.sort_values("event_id", inplace=True)
        df.reset_index(drop=True, inplace=True)

    same_ids = b.event_id.equals(p.event_id) and b.event_id.equals(k.event_id)
    if not same_ids: raise RuntimeError("Event ID universes differ")
    same_conv = np.array_equal(b.conv_id, p.conv_id) and np.array_equal(b.conv_id, k.conv_id)
    same_target = np.array_equal(b.target_movie_id,p.target_movie_id) and np.array_equal(b.target_movie_id,k.target_movie_id)
    if len(b) != N_EVENTS or b.conv_id.nunique() != N_CONV:
        raise RuntimeError(f"Expected {N_EVENTS} events / {N_CONV} conversations; got {len(b)} / {b.conv_id.nunique()}")
    if not same_conv or not same_target:
        raise RuntimeError("Conversation or target mapping differs across baseline/proposed/KBRD")

    # Conversation cluster sizes.
    clusters = b.groupby("conv_id").agg(
        events=("event_id","size"), first_event_id=("event_id","min"),
        last_event_id=("event_id","max")).reset_index()
    clusters["event_share_percent"] = 100 * clusters.events / len(b)
    freq = clusters.events.value_counts().sort_index()
    cluster_summary = {
        "events":len(b), "conversations":int(b.conv_id.nunique()),
        "mean_events_per_conversation":float(clusters.events.mean()),
        "median_events_per_conversation":float(clusters.events.median()),
        "sd_events_per_conversation":float(clusters.events.std(ddof=1)),
        "min_events_per_conversation":int(clusters.events.min()),
        "max_events_per_conversation":int(clusters.events.max()),
        "single_event_conversations":int((clusters.events==1).sum()),
        "multi_event_conversations":int((clusters.events>1).sum()),
        "event_count_frequency":{str(int(x)):int(y) for x,y in freq.items()}
    }

    # Verify top-10/group/candidate lengths and group labels against catalog.
    align = []
    for i in range(len(b)):
        br, pr, kr = b.iloc[i], p.iloc[i], k.iloc[i]
        bi, pi, ki = ids(br.top10_movie_ids), ids(pr.top10_movie_ids), ids(kr.top_movie_ids)
        bg, pg = ids(br.top10_group_ids), ids(pr.top10_group_ids)
        align.append({
            "event_id":int(br.event_id), "conv_id":int(br.conv_id),
            "baseline_top10_len":len(bi), "proposed_top10_len":len(pi),
            "kbrd_top500_len":len(ki),
            "baseline_group_labels_match":bg==[int(item2group[x]) for x in bi],
            "proposed_group_labels_match":pg==[int(item2group[x]) for x in pi],
            "baseline_proposed_top10_identical":bi==pi
        })
    align = pd.DataFrame(align)
    if not (align.baseline_top10_len.eq(10).all() and align.proposed_top10_len.eq(10).all()
            and align.kbrd_top500_len.eq(500).all()
            and align.baseline_group_labels_match.all()
            and align.proposed_group_labels_match.all()):
        raise RuntimeError("Top-K lengths or popularity-group labels failed validation")

    # Rebuild the cell universe from every frozen Top-500 candidate opportunity.
    cells_set = set()
    for value in k.top_movie_ids:
        for mid in ids(value):
            for cat in categories.get(int(mid), []):
                cells_set.add((cat, int(item2group[int(mid)])))
    cells = sorted(cells_set)
    if len(cells) != 112:
        raise RuntimeError(f"Expected 112 category-group cells; got {len(cells)}")

    # Reuse the exact contribution implementation from the frozen 22.24C script.
    b_ids,b_conv,bh,bn,be,bo = inf.build_event_contributions(k,b,item2group,categories,cells,ref)
    p_ids,p_conv,ph,pn,pe,po = inf.build_event_contributions(k,p,item2group,categories,cells,ref)
    if b_ids != p_ids or not np.array_equal(b_conv,p_conv):
        raise RuntimeError("Contribution event/conversation alignment failed")
    if not np.allclose(bo,po,rtol=1e-10,atol=1e-12):
        raise RuntimeError("Baseline and proposed opportunity contributions differ")

    total_opp = bo.sum(axis=0)
    be_total, pe_total = be.sum(axis=0), pe.sum(axis=0)
    all_mask = total_opp > EPS

    # Distinct-conversation support is based on whether a conversation contributes
    # positive opportunity to a cell (not the number of events in that conversation).
    conv_ids = np.asarray(sorted(b.conv_id.unique()), dtype=np.int64)
    conv_idx = {int(x):i for i,x in enumerate(conv_ids)}
    event_ci = np.asarray([conv_idx[int(x)] for x in b_conv])
    cluster_opp = np.zeros((len(conv_ids),len(cells)),dtype=float)
    cluster_be = np.zeros_like(cluster_opp)
    cluster_pe = np.zeros_like(cluster_opp)
    np.add.at(cluster_opp,event_ci,bo)
    np.add.at(cluster_be,event_ci,be)
    np.add.at(cluster_pe,event_ci,pe)
    support = (cluster_opp > EPS).sum(axis=0)
    support_mask = support >= MIN_SUPPORT

    # Reproduce the original 22.16 ledger calculations independently.
    lb, lp = ref.ExposureLedger(), ref.ExposureLedger()
    k_by = k.set_index("event_id"); b_by = b.set_index("event_id"); p_by = p.set_index("event_id")
    for eid in b_ids:
        kr,br,pr = k_by.loc[eid],b_by.loc[eid],p_by.loc[eid]
        cand = np.asarray(ids(kr.top_movie_ids),dtype=np.int64)
        cg = np.asarray([item2group[int(x)] for x in cand],dtype=int)
        bid = np.asarray(ids(br.top10_movie_ids),dtype=np.int64)
        bg = np.asarray(ids(br.top10_group_ids),dtype=int)
        pid = np.asarray(ids(pr.top10_movie_ids),dtype=np.int64)
        pg = np.asarray(ids(pr.top10_group_ids),dtype=int)
        lb.update(cand,cg,bid,bg,categories)
        lp.update(cand,cg,pid,pg,categories)

    ledger_b,ledger_p = float(ref.fairness_mad(lb)),float(ref.fairness_mad(lp))
    full_b,full_p = mad(be_total,total_opp,all_mask),mad(pe_total,total_opp,all_mask)
    filt_b,filt_p = mad(be_total,total_opp,support_mask),mad(pe_total,total_opp,support_mask)

    # Saved 22.24B aggregate metrics.
    ag = agg.set_index("method")
    bm,pm = "A0_KBRD_baseline","proposed_context_kg_gated"
    for m in (bm,pm):
        if m not in ag.index: raise ValueError(f"Missing method {m} in aggregate_metrics.csv")
    saved_b,saved_p = float(ag.loc[bm,"category_group_log_rate_MAD"]),float(ag.loc[pm,"category_group_log_rate_MAD"])

    utility = {
        "baseline_HR10_reconstructed":float(b.hit10.mean()),
        "baseline_HR10_saved":float(ag.loc[bm,"HR@10"]),
        "proposed_HR10_reconstructed":float(p.hit10.mean()),
        "proposed_HR10_saved":float(ag.loc[pm,"HR@10"]),
        "baseline_NDCG10_reconstructed":float(b.ndcg10.mean()),
        "baseline_NDCG10_saved":float(ag.loc[bm,"NDCG@10"]),
        "proposed_NDCG10_reconstructed":float(p.ndcg10.mean()),
        "proposed_NDCG10_saved":float(ag.loc[pm,"NDCG@10"])
    }
    utility["all_utility_metrics_match"] = all([
        close(utility["baseline_HR10_reconstructed"],utility["baseline_HR10_saved"]),
        close(utility["proposed_HR10_reconstructed"],utility["proposed_HR10_saved"]),
        close(utility["baseline_NDCG10_reconstructed"],utility["baseline_NDCG10_saved"]),
        close(utility["proposed_NDCG10_reconstructed"],utility["proposed_NDCG10_saved"])
    ])

    fairness = pd.DataFrame([
      {"estimand":"22.24B_all_positive_opportunity_cells","baseline_reconstructed_MAD":full_b,
       "proposed_reconstructed_MAD":full_p,"baseline_22_16_ledger_MAD":ledger_b,
       "proposed_22_16_ledger_MAD":ledger_p,"baseline_saved_22_24B_MAD":saved_b,
       "proposed_saved_22_24B_MAD":saved_p,"cells":int(all_mask.sum()),
       "fairness_reduction_percent":100*(full_b-full_p)/full_b,
       "vectorized_matches_ledger":close(full_b,ledger_b) and close(full_p,ledger_p),
       "ledger_matches_saved_22_24B":close(ledger_b,saved_b) and close(ledger_p,saved_p)},
      {"estimand":"22.24C_fixed_cells_support_ge_20_conversations","baseline_reconstructed_MAD":filt_b,
       "proposed_reconstructed_MAD":filt_p,"baseline_22_16_ledger_MAD":np.nan,
       "proposed_22_16_ledger_MAD":np.nan,"baseline_saved_22_24B_MAD":np.nan,
       "proposed_saved_22_24B_MAD":np.nan,"cells":int(support_mask.sum()),
       "fairness_reduction_percent":100*(filt_b-filt_p)/filt_b,
       "vectorized_matches_ledger":np.nan,"ledger_matches_saved_22_24B":np.nan}
    ])

    cell_df = pd.DataFrame({
      "category":[x[0] for x in cells],"popularity_group":[int(x[1]) for x in cells],
      "supporting_conversations":support,"total_conversations":len(conv_ids),
      "included_at_20_conversations":support_mask,
      "opportunity":total_opp,"baseline_exposure":be_total,"proposed_exposure":pe_total
    })

    checks = {
      "events_4003":len(b)==N_EVENTS,"conversations_1283":b.conv_id.nunique()==N_CONV,
      "event_ids_match":bool(same_ids),"conversation_mapping_matches":bool(same_conv),
      "target_mapping_matches":bool(same_target),
      "all_top10_and_top500_lengths_valid":bool(align.baseline_top10_len.eq(10).all() and align.proposed_top10_len.eq(10).all() and align.kbrd_top500_len.eq(500).all()),
      "all_group_labels_match_catalog":bool(align.baseline_group_labels_match.all() and align.proposed_group_labels_match.all()),
      "same_opportunity_contributions":bool(np.allclose(bo,po,rtol=1e-10,atol=1e-12)),
      "112_cells_reconstructed":len(cells)==112,"106_cells_meet_support_20":int(support_mask.sum())==106,
      "22_16_ledger_parity":close(full_b,ledger_b) and close(full_p,ledger_p),
      "22_24B_fairness_reconciliation":close(ledger_b,saved_b) and close(ledger_p,saved_p),
      "22_24B_utility_reconciliation":utility["all_utility_metrics_match"],
      "training_performed":False,"retrieval_rerun":False,"ranking_policy_modified":False,
      "independent_test_rerun":False
    }
    checks["overall_pass"] = all(v is True for k,v in checks.items() if k not in
      {"training_performed","retrieval_rerun","ranking_policy_modified","independent_test_rerun"})

    a.out.mkdir(parents=True,exist_ok=True)
    clusters.to_csv(a.out/"conversation_cluster_sizes.csv",index=False)
    align.to_csv(a.out/"event_alignment.csv",index=False)
    cell_df.to_csv(a.out/"fairness_cell_reconstruction.csv",index=False)
    fairness.to_csv(a.out/"fairness_reconciliation.csv",index=False)
    summary = {
      "experiment":"22.24C_event_level_reconstruction_audit","diagnostic_only":True,
      "cluster_size_summary":cluster_summary,"fairness_cells_total":len(cells),
      "fairness_cells_included_at_support_20":int(support_mask.sum()),
      "fairness_cells_excluded":int((~support_mask).sum()),
      "fairness_reconciliation":fairness.to_dict(orient="records"),
      "utility_reconciliation":utility,"validation":checks,
      "interpretation":{
        "22_24B":"MAD across all category-group cells with positive Top-500 opportunity.",
        "22_24C":"MAD across the fixed subset supported by at least 20 distinct conversations.",
        "warning":"22.24B and 22.24C fairness MAD values are different estimands.",
        "scope":"This reconstructs point estimates and cluster structure; it does not compute a new null-test p-value or rerun the bootstrap."
      }
    }
    (a.out/"audit_summary.json").write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False),encoding="utf-8")
    print("\nCLUSTER SIZE SUMMARY\n",json.dumps(cluster_summary,indent=2))
    print("\nFAIRNESS RECONCILIATION\n",fairness.to_string(index=False))
    print("\nUTILITY RECONCILIATION\n",json.dumps(utility,indent=2))
    print("\nVALIDATION\n",json.dumps(checks,indent=2))
    print("\nOutput:",a.out.resolve())
    if not checks["overall_pass"]: raise SystemExit("AUDIT FAILED — inspect audit_summary.json")
    print("\nAUDIT PASS — original scripts and outputs were not modified.")

if __name__ == "__main__":
    main()



