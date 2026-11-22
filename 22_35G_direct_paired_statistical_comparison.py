#!/usr/bin/env python
"""22.35G: paired conversation-cluster bootstrap for frozen 22.35F R0 slates.
No training, retrieval, reranking, tuning, checkpoint edits, or test-set access.
"""
import argparse, hashlib, importlib.util, json, platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

SYSTEMS = ("A0_KGSF", "A1_22_16", "A2_22_18", "A3_KG_FairCRS")
COMPS = SYSTEMS[:3]
METRICS = ("HR@10", "NDCG@10", "MRR@10", "L_fair")
EPS, K = 1e-12, 10

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return tuple(map(float, np.percentile(x, [2.5, 97.5]))) if len(x) else (np.nan, np.nan)

def tail_p(x):
    # Descriptive bootstrap tail probability, not an exact randomization-test p-value.
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return np.nan
    return float(min(1., 2 * min((np.sum(x <= 0)+1)/(len(x)+1),
                                 (np.sum(x >= 0)+1)/(len(x)+1))))

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty(len(p)); run = 0.
    for rank, ix in enumerate(order):
        run = max(run, min(1., (len(p)-rank)*p[ix])); out[ix] = run
    return out

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-level", type=Path, default=Path(
        "results_corrected/experiment_22_35/validated_fairness_evaluation/event_level_system_comparisons.csv"))
    ap.add_argument("--decisions", type=Path, default=Path(
        "results_corrected/experiment_22_35/validated_fairness_evaluation/parameter_decisions.json"))
    ap.add_argument("--kgsf", type=Path, default=Path(
        "results_corrected/experiment_22_35/kgsf_validation/kgsf_validation_top500.csv"))
    ap.add_argument("--q-r0", type=Path, default=Path(
        "results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R0_filtered.csv"))
    ap.add_argument("--catalog", type=Path, default=Path("catalog_audit_corrected.pkl"))
    ap.add_argument("--profiles", type=Path, default=Path(
        "results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"))
    ap.add_argument("--ref16", type=Path, default=Path("22_16_category_context_utility_calibration.py"))
    ap.add_argument("--f-script", type=Path, default=Path("22_35F_validated_fairness_evaluation.py"))
    ap.add_argument("--out", type=Path, default=Path(
        "results_corrected/experiment_22_35/direct_paired_statistical_comparison"))
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=22357)
    a = ap.parse_args()
    if a.reps < 1000: raise ValueError("Use at least 1,000 replicates (default 10,000).")
    inputs = [a.event_level,a.decisions,a.kgsf,a.q_r0,a.catalog,a.profiles,a.ref16,a.f_script]
    for p in inputs:
        if not p.is_file(): raise FileNotFoundError(p)
    if a.out.exists(): raise FileExistsError(f"Refusing to overwrite existing output: {a.out}")

    # Import deterministic 22.35F helpers only. F.main() is never called.
    F = module(a.f_script, "frozen_2235f_helpers")
    R = module(a.ref16, "frozen_2216_helpers")
    item2group, catalog, categories = R.load_metadata(a.catalog, a.profiles)
    events, cands = F.load_data(a.kgsf, a.q_r0, catalog)
    if len(events) != 3010 or len({v["conv"] for v in events.values()}) != 883:
        raise RuntimeError("Expected the validated 3,010 events / 883 conversations.")
    d = json.loads(a.decisions.read_text(encoding="utf-8"))
    qthr, rmin = float(d["A3_KG_FairCRS"]["parameters"]["q_threshold"]), float(d["A3_KG_FairCRS"]["parameters"]["R_min"])
    wc, tc, hc = R.split_conversations([v["conv"] for v in events.values()])
    if wc & tc or wc & hc or tc & hc: raise RuntimeError("Conversation splits overlap.")
    warm = sorted(e for e,v in events.items() if v["conv"] in wc)
    tune = sorted(e for e,v in events.items() if v["conv"] in tc)
    held = sorted(e for e,v in events.items() if v["conv"] in hc)

    df = pd.read_csv(a.event_level)
    required = {"system","event_id","conv_id","hit10","ndcg10","mrr10","top10_movie_ids","split","variant"}
    if required-set(df): raise RuntimeError(f"Event CSV missing columns: {sorted(required-set(df))}")
    df = df[(df.variant=="R0_filtered") & df.system.isin(SYSTEMS)].copy()
    df["event_id"] = df.event_id.astype(int); df["conv_id"] = df.conv_id.astype(int)
    lookup = {}
    for system in SYSTEMS:
        for split, ids in (("tune",tune),("heldout",held)):
            z = df[(df.system==system)&(df.split==split)].copy()
            if z.event_id.duplicated().any() or set(z.event_id)!=set(ids):
                raise RuntimeError(f"{system}/{split} event IDs differ from frozen split.")
            if any(int(row.conv_id)!=events[int(row.event_id)]["conv"] for row in z.itertuples()):
                raise RuntimeError(f"{system}/{split} conversation mapping mismatch.")
            lookup[system,split] = z.set_index("event_id", drop=False)

    def slate(system, event_id, split):
        x = F.parse_list(lookup[system,split].loc[event_id,"top10_movie_ids"],"top10_movie_ids")
        x = list(map(int,x))
        if len(x)!=K or len(set(x))!=K: raise RuntimeError(f"Invalid top-10 slate: {system}/{event_id}")
        return x

    # Warm history is KGSF for all systems; tuning history uses each system's frozen slate.
    warm_state = F.warm_prop(warm,cands,item2group,categories,qthr)
    initial = {}
    for system in SYSTEMS:
        state = warm_state.clone()
        for e in tune:
            state.add_R(cands[e],item2group,categories,qthr)
            state.add_E(slate(system,e,"tune"),item2group,categories)
        fair,n,_ = F.proposal_terms(state,rmin)
        if not np.isfinite(fair) or n<1: raise RuntimeError(f"{system}: invalid cumulative warm+tune fairness.")
        initial[system] = state

    byconv = defaultdict(list)
    for e in held: byconv[int(events[e]["conv"])].append(e)
    convs = np.asarray(sorted(byconv),dtype=np.int64)
    if len(convs)!=len(hc): raise RuntimeError("Held-out conversation count mismatch.")
    utility = {}; slates = {}
    for system in SYSTEMS:
        utility[system] = {}; slates[system] = {}
        for e in held:
            row = lookup[system,"heldout"].loc[e]
            utility[system][e] = np.asarray([row.hit10,row.ndcg10,row.mrr10],float)
            slates[system][e] = slate(system,e,"heldout")

    rng = np.random.default_rng(a.seed)
    # Arrays store A3 minus each comparator, with paired identical cluster draws.
    deltas = {c:{m:np.empty(a.reps) for m in METRICS} for c in COMPS}
    absolute = {s:{m:np.empty(a.reps) for m in METRICS} for s in SYSTEMS}
    for b in range(a.reps):
        draw = rng.choice(convs,size=len(convs),replace=True)
        sampled = [e for c in draw for e in byconv[int(c)]]  # preserve repeated-cluster multiplicity
        current = {}
        for system in SYSTEMS:
            u = np.vstack([utility[system][e] for e in sampled])
            state = initial[system].clone()
            for e in sampled:
                state.add_R(cands[e],item2group,categories,qthr)
                state.add_E(slates[system][e],item2group,categories)
            fair,_,_ = F.proposal_terms(state,rmin)
            current[system] = {"HR@10":float(u[:,0].mean()),"NDCG@10":float(u[:,1].mean()),
                               "MRR@10":float(u[:,2].mean()),"L_fair":float(fair)}
            for m in METRICS: absolute[system][m][b] = current[system][m]
        for comp in COMPS:
            for m in METRICS: deltas[comp][m][b] = current["A3_KG_FairCRS"][m]-current[comp][m]
        if (b+1)%1000==0: print(f"Bootstrap {b+1:,}/{a.reps:,}")

    rows=[]
    for comp in COMPS:
        for m in METRICS:
            x=deltas[comp][m]; lo,hi=ci(x)
            rows.append({"comparison":f"A3_KG_FairCRS_minus_{comp}","comparator":comp,"metric":m,
                "mean_bootstrap_delta":float(np.mean(x)),"ci95_low":lo,"ci95_high":hi,
                "bootstrap_two_sided_tail_probability":tail_p(x),
                "beneficial_direction":"lower" if m=="L_fair" else "higher",
                "ci_excludes_zero":bool(hi<0 or lo>0),"bootstrap_reps":a.reps,"cluster_unit":"conversation"})
    contrasts=pd.DataFrame(rows)
    contrasts["holm_adjusted_bootstrap_tail_probability"]=np.nan
    for metric_set in (("L_fair",),("HR@10","NDCG@10","MRR@10")):
        mask=contrasts.metric.isin(metric_set)
        contrasts.loc[mask,"holm_adjusted_bootstrap_tail_probability"]=holm(
            contrasts.loc[mask,"bootstrap_two_sided_tail_probability"].to_numpy())

    absrows=[]
    for system in SYSTEMS:
        for m in METRICS:
            lo,hi=ci(absolute[system][m])
            absrows.append({"system":system,"metric":m,"bootstrap_mean":float(np.mean(absolute[system][m])),
                            "ci95_low":lo,"ci95_high":hi,"bootstrap_reps":a.reps})
    absdf=pd.DataFrame(absrows)

    obs=[]
    for system in SYSTEMS:
        state=initial[system].clone()
        for e in held:
            state.add_R(cands[e],item2group,categories,qthr)
            state.add_E(slates[system][e],item2group,categories)
        fair,n,_=F.proposal_terms(state,rmin); z=lookup[system,"heldout"]
        obs.append({"system":system,"events":len(held),"conversations":len(convs),
                    "HR@10":float(z.hit10.mean()),"NDCG@10":float(z.ndcg10.mean()),
                    "MRR@10":float(z.mrr10.mean()),"L_fair_cumulative":fair,
                    "active_categories_cumulative":n})
    obsdf=pd.DataFrame(obs)

    a.out.mkdir(parents=True,exist_ok=False)
    contrasts.to_csv(a.out/"paired_contrast_summary.csv",index=False)
    absdf.to_csv(a.out/"absolute_bootstrap_summary.csv",index=False)
    obsdf.to_csv(a.out/"observed_heldout_system_summary.csv",index=False)
    # Full replicate deltas are retained for reproducibility and independent checks.
    rep=[]
    for comp in COMPS:
        for m in METRICS:
            rep.extend({"replicate":i+1,"comparator":comp,"metric":m,"A3_minus_comparator":float(v)}
                       for i,v in enumerate(deltas[comp][m]))
    pd.DataFrame(rep).to_csv(a.out/"paired_bootstrap_replicates.csv",index=False)

    checks={"experiment":"22.35G","validation_only":True,"independent_test_read":False,
      "training_performed":False,"retrieval_or_reranking_performed":False,"22_35F_outputs_modified":False,
      "same_heldout_events_all_systems":True,"same_cluster_draws_all_systems":True,
      "cluster_multiplicity_preserved":True,"fairness_recomputed_from_R_E_each_replicate":True,
      "proposal_objective_reused":True,"system_specific_warm_plus_tune_initial_state":True,
      "primary_variant":"R0_filtered","bootstrap_reps":a.reps,"seed":a.seed,
      "all_contrasts_finite":bool(np.isfinite(contrasts.mean_bootstrap_delta).all()),
      "all_fairness_intervals_finite":bool(np.isfinite(contrasts[contrasts.metric=="L_fair"][["ci95_low","ci95_high"]].to_numpy()).all())}
    checks["overall_pass"]=all(checks[k] for k in ("same_heldout_events_all_systems","same_cluster_draws_all_systems",
      "cluster_multiplicity_preserved","fairness_recomputed_from_R_E_each_replicate","all_contrasts_finite",
      "all_fairness_intervals_finite"))
    (a.out/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    files={"event_level":a.event_level,"decisions":a.decisions,"kgsf":a.kgsf,"q_r0":a.q_r0,
           "catalog":a.catalog,"profiles":a.profiles,"ref16":a.ref16,"f_script":a.f_script}
    manifest={"experiment":"22.35G_direct_paired_statistical_comparison",
      "created_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
      "numpy":np.__version__,"pandas":pd.__version__,"scope":"frozen_22.35F_validation_slates_R0",
      "parameters_reused":{"q_threshold":qthr,"R_min":rmin},
      "bootstrap":{"method":"paired conversation-cluster percentile bootstrap","reps":a.reps,
        "seed":a.seed,"unit":"held-out conversation","same_draws_across_systems":True,
        "fixed_slates":True,"multiplicity_preserved":True},
      "multiplicity":{"fairness":"Holm across 3 prespecified contrasts",
        "utility":"Holm across 9 contrasts (3 metrics x 3 comparators)"},
      "p_value_interpretation":"Bootstrap tail probabilities are descriptive, not exact randomization-test p-values; interpret with paired percentile CIs.",
      "no_training":True,"no_reranking":True,"no_retrieval":True,"no_checkpoint_changes":True,
      "independent_test_read":False,"inputs":{k:{"path":str(v),"sha256":digest(v)} for k,v in files.items()},
      "outputs":["paired_contrast_summary.csv","absolute_bootstrap_summary.csv",
        "observed_heldout_system_summary.csv","paired_bootstrap_replicates.csv","validation_checks.json","experiment_manifest.json"],
      "validation_checks":checks}
    (a.out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\n22.35G COMPLETE\n",obsdf.to_string(index=False),"\n",contrasts.to_string(index=False))
    print("Validation:",checks["overall_pass"],"| Independent test: NOT READ | 22.35F untouched")
    if not checks["overall_pass"]: raise RuntimeError("22.35G checks failed; inspect validation_checks.json")

if __name__=="__main__": main()
