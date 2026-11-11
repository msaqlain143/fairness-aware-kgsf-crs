#!/usr/bin/env python
"""22.11B: extend frozen validation diagnostics; no training/retrieval/test use."""
import argparse, ast, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

BASE = "B0_KBRD"
METHODS = ["B_category_only_lambda_0p2", "B_KG_gated_lambda_0p2",
           "B_category_only_lambda_0p4", "B_KG_gated_lambda_0p4"]
SEED = 22111

def ids(x):
    if isinstance(x, (list, tuple, np.ndarray)): return [int(v) for v in x]
    try: return [int(v) for v in json.loads(str(x))]
    except Exception: return [int(v) for v in ast.literal_eval(str(x))]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alignment", type=Path, default=Path("results_corrected/experiment_22_11/candidate_level_kbrd_kg_alignment.csv"))
    p.add_argument("--events", type=Path, default=Path("results_corrected/experiment_22_10/event_level_results.csv"))
    p.add_argument("--catalog-pickle", type=Path, default=Path("catalog_audit_corrected.pkl"))
    p.add_argument("--out", type=Path, default=Path("results_corrected/experiment_22_11"))
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    if a.bootstrap < 100: raise ValueError("--bootstrap must be >=100")

    c = pd.read_csv(a.alignment)
    e = pd.read_csv(a.events)
    with a.catalog_pickle.open("rb") as f: cat = pickle.load(f)
    needc = {"event_id","conv_id","movie_id","kbrd_rank","kg_rank","kg_supported"}
    neede = {"event_id","conv_id","method","top10_movie_ids"}
    if not needc.issubset(c) or not neede.issubset(e): raise ValueError("Input schema mismatch.")
    for x in ["event_id","conv_id","movie_id","kg_rank","kg_supported"]: c[x] = pd.to_numeric(c[x]).astype(int)
    for x in ["event_id","conv_id"]: e[x] = pd.to_numeric(e[x]).astype(int)
    if c.duplicated(["event_id","movie_id"]).any(): raise ValueError("Duplicate event/movie alignment.")
    if e.duplicated(["event_id","method"]).any(): raise ValueError("Duplicate event/method rows.")
    if set(e.method) != {BASE,*METHODS}: raise ValueError("Unexpected method set.")
    mapping = {int(k):int(v) for k,v in cat["item2group"].items()}
    c["popularity_group"] = c.movie_id.map(mapping)
    if c.popularity_group.isna().any(): raise ValueError("Missing frozen popularity-group assignment.")
    c["popularity_group"] = c.popularity_group.astype(int)

    # Coverage and KG rank by the frozen five-group catalog partition.
    coverage = c.groupby("popularity_group").agg(
        candidate_rows=("movie_id","size"), unique_movies=("movie_id","nunique"),
        events=("event_id","nunique"), supported_rows=("kg_supported","sum"),
        kg_support_coverage=("kg_supported","mean")).reset_index()
    supported = c[c.kg_supported == 1]
    rank_summary = supported.groupby("popularity_group").kg_rank.agg(
        supported_rows="size", mean_kg_rank="mean", median_kg_rank="median",
        q25_kg_rank=lambda x:x.quantile(.25), q75_kg_rank=lambda x:x.quantile(.75),
        q90_kg_rank=lambda x:x.quantile(.90)).reset_index()

    lookup = c.set_index(["event_id","movie_id"])[["conv_id","kg_rank","kg_supported","popularity_group"]]
    top = {}
    for r in e.itertuples(index=False):
        top[(int(r.event_id),str(r.method))] = (int(r.conv_id), ids(r.top10_movie_ids))
    evs = sorted(e.event_id.unique())
    for event in evs:
        if len(top[(int(event),BASE)][1]) != 10: raise ValueError(f"Invalid baseline Top-10 at {event}")
    promotions, demotions, eventrows = [], [], []
    for method in METHODS:
        for event in evs:
            conv,b = top[(int(event),BASE)]
            conv2,m = top[(int(event),method)]
            if conv != conv2: raise ValueError(f"Conversation mismatch at event {event}")
            entered, left = sorted(set(m)-set(b)), sorted(set(b)-set(m))
            if len(entered) != len(left): raise ValueError("Top-10 set-size mismatch.")
            for movie, dest in [(x,promotions) for x in entered] + [(x,demotions) for x in left]:
                if (int(event),int(movie)) not in lookup.index: raise ValueError(f"Movie not in candidate pool: {event}/{movie}")
                z = lookup.loc[(int(event),int(movie))]
                dest.append(dict(event_id=int(event),conv_id=conv,method=method,movie_id=int(movie),
                    kg_rank=int(z.kg_rank),kg_supported=int(z.kg_supported),
                    popularity_group=int(z.popularity_group)))
            ps = [int(lookup.loc[(int(event),x)].kg_supported) for x in entered]
            ds = [int(lookup.loc[(int(event),x)].kg_supported) for x in left]
            eventrows.append(dict(event_id=int(event),conv_id=conv,method=method,
                promoted_count=len(entered),demoted_count=len(left),
                promoted_supported=sum(ps),demoted_supported=sum(ds)))

    pro, dem, ev = pd.DataFrame(promotions), pd.DataFrame(demotions), pd.DataFrame(eventrows)
    summaries=[]
    for method in METHODS:
        x,y=pro[pro.method==method],dem[dem.method==method]
        pr,dr=float(x.kg_supported.mean()),float(y.kg_supported.mean())
        summaries.append(dict(method=method,promoted=len(x),demoted=len(y),
            promoted_supported=int(x.kg_supported.sum()),demoted_supported=int(y.kg_supported.sum()),
            promoted_support_rate=pr,demoted_support_rate=dr,
            promoted_minus_demoted_support_rate=pr-dr,
            promoted_supported_mean_kg_rank=float(x.loc[x.kg_supported==1,"kg_rank"].mean()) if x.kg_supported.any() else np.nan,
            demoted_supported_mean_kg_rank=float(y.loc[y.kg_supported==1,"kg_rank"].mean()) if y.kg_supported.any() else np.nan))
    summary=pd.DataFrame(summaries)

    # Cluster bootstrap: resample conversations, aggregating all events in each draw.
    convs=np.array(sorted(ev.conv_id.unique()))
    per={}
    for method in METHODS:
        per[method]=ev[ev.method==method].groupby("conv_id").agg(
            ps=("promoted_supported","sum"), pn=("promoted_count","sum"),
            ds=("demoted_supported","sum"), dn=("demoted_count","sum")).reindex(convs,fill_value=0)
    rng=np.random.default_rng(SEED); boot={m:[] for m in METHODS}
    for _ in range(a.bootstrap):
        draw=rng.choice(convs,size=len(convs),replace=True)
        for m in METHODS:
            z=per[m].loc[draw]; pn=z.pn.sum(); dn=z.dn.sum()
            if pn and dn: boot[m].append(z.ps.sum()/pn-z.ds.sum()/dn)
    br=[]
    for m in METHODS:
        v=np.asarray(boot[m]); point=float(summary.loc[summary.method==m,"promoted_minus_demoted_support_rate"].iloc[0])
        br.append(dict(method=m,point_difference=point,ci95_low=np.quantile(v,.025),
            ci95_high=np.quantile(v,.975),bootstrap_probability_positive=np.mean(v>0),
            replicates=len(v),conversation_clusters=len(convs),sampling_unit="conversation_id",
            interpretation="association only; not causal"))
    bootstrap=pd.DataFrame(br)

    a.out.mkdir(parents=True,exist_ok=True)
    outputs={
      "popularity_group_kg_coverage.csv":coverage,
      "popularity_group_kg_rank_summary.csv":rank_summary,
      "top10_promotions.csv":pro,"top10_demotions.csv":dem,
      "top10_movement_summary.csv":summary,"top10_movement_bootstrap.csv":bootstrap}
    for name,df in outputs.items(): df.to_csv(a.out/name,index=False)
    checks=dict(candidate_event_movie_unique=not c.duplicated(["event_id","movie_id"]).any(),
      frozen_popularity_mapping_complete=bool(c.popularity_group.notna().all()),
      groups_are_0_to_4=set(c.popularity_group.unique())=={0,1,2,3,4},
      five_methods_and_baseline_present=set(e.method)=={BASE,*METHODS},
      all_methods_share_event_ids=all(set(e[e.method==m].event_id)==set(e[e.method==BASE].event_id) for m in METHODS),
      promotion_and_demotion_counts_equal=bool((ev.promoted_count==ev.demoted_count).all()),
      test_split_used=False,training_performed=False,retrieval_rerun=False,new_coefficients=False)
    #checks["overall_pass"]=all(checks.values())
    integrity_keys = [
    "candidate_event_movie_unique",
    "frozen_popularity_mapping_complete",
    "groups_are_0_to_4",
    "five_methods_and_baseline_present",
    "all_methods_share_event_ids",
    "promotion_and_demotion_counts_equal",
    ]

    guardrail_keys = [
    "test_split_used",
    "training_performed",
    "retrieval_rerun",
    "new_coefficients",
    ]

    checks["overall_pass"] = (
    all(checks[k] for k in integrity_keys)
    and all(not checks[k] for k in guardrail_keys)
    )
    (a.out/"extension_validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    manifest=dict(experiment="22.11B_existing_KG_evidence_diagnostic_extension",
      scope="frozen validation exports only",popularity_mapping="catalog_audit_corrected.pkl:item2group",
      promotion="method Top-10 minus B0_KBRD Top-10",
      demotion="B0_KBRD Top-10 minus method Top-10",
      bootstrap=dict(replicates=a.bootstrap,seed=SEED,sampling_unit="conversation_id",
                     estimand="promoted KG-support rate minus demoted KG-support rate"),
      no_training=True,no_retrieval_rerun=True,no_test_evaluation=True,outputs=list(outputs)+["extension_validation_checks.json"])
    (a.out/"extension_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\nPopularity-group KG coverage:\n",coverage.to_string(index=False))
    print("\nTop-10 movement:\n",summary.to_string(index=False))
    print("\nCluster bootstrap:\n",bootstrap.to_string(index=False))
    print("\nValidation:", "PASS" if checks["overall_pass"] else "FAIL")
    if not checks["overall_pass"]: raise SystemExit(1)

if __name__=="__main__": main()
