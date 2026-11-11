#!/usr/bin/env python
"""Experiment 22.9: KBRD cross-backbone validation of KG-FairCRS.

B0 = original KBRD; B1 = adaptive category calibration; B2 = KG-gated
adaptive category calibration. Validation only; no test split is accessed.
"""
from __future__ import annotations
import argparse, ast, json, math, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SEED, BOOT_SEED, TOP_K, CANDIDATE_K, EPS = 2200, 22900, 10, 500, 1e-12
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20]

def as_list(x, field):
    try:
        v = json.loads(x) if isinstance(x, str) else x
    except Exception:
        v = ast.literal_eval(x)
    if not isinstance(v, list):
        raise ValueError(f"{field} must be a list")
    return v

def as_bool(x):
    if pd.isna(x): return False
    if isinstance(x, (bool, np.bool_)): return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes"}

def genres(x, delim):
    if pd.isna(x): return []
    return sorted({" ".join(t.strip().casefold().split())
                   for t in str(x).split(delim) if t.strip()})

def split_conversations(conv_ids):
    u = np.asarray(sorted(set(map(int, conv_ids))), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    rng.shuffle(u)
    a, b = round(.60*len(u)), round(.20*len(u))
    return set(map(int,u[:a])), set(map(int,u[a:a+b])), set(map(int,u[a+b:]))

def load_metadata(catalog_path, profile_path):
    with open(catalog_path, "rb") as f: meta = pickle.load(f)
    item2group = {int(k):int(v) for k,v in meta["item2group"].items()}
    catalog = set(map(int, meta["movie_ids"]))
    if set(item2group) != catalog or any(v not in range(5) for v in item2group.values()):
        raise ValueError("Catalog/item2group alignment or group IDs invalid")
    p = pd.read_csv(profile_path)
    needed = {"movie_entity_id","canonical_genres","imdb_genres",
              "imdb_genre_evidence","imdb_conflicting_records",
              "accepted_genre_evidence","identity_resolution_status"}
    if needed - set(p.columns): raise ValueError(f"Profile columns missing: {needed-set(p.columns)}")
    p["movie_entity_id"] = p.movie_entity_id.astype(int)
    if p.movie_entity_id.duplicated().any(): raise ValueError("Duplicate category profile IDs")
    cats = {m:[] for m in catalog}
    for r in p.itertuples(index=False):
        m = int(r.movie_entity_id)
        if m not in catalog or str(r.identity_resolution_status).upper()=="IDENTITY_REVIEW_REQUIRED":
            continue
        accepted=[]
        if as_bool(r.imdb_genre_evidence) and not as_bool(r.imdb_conflicting_records):
            raw = r.imdb_genres
            if not pd.isna(raw): accepted += genres(raw, "|" if "|" in str(raw) else ";")
        if as_bool(r.accepted_genre_evidence):
            accepted += genres(r.canonical_genres, ";")
        cats[m] = sorted(set(accepted))
    return meta,item2group,catalog,cats,p

class Ledger:
    def __init__(self):
        self.exp=defaultdict(float); self.opp=defaultdict(float)
    @staticmethod
    def opp_weights(n):
        w=1/np.log2(np.arange(1,n+1,dtype=float)+1)
        return w/(w.sum()+EPS)
    def update(self,ids,cats,groups):
        ow=self.opp_weights(len(ids))
        for j,m in enumerate(ids):
            cs=cats.get(int(m),[])
            for c in cs: self.opp[c,int(groups[j])] += float(ow[j])/len(cs)
        for pos,m in enumerate(ids[:TOP_K]):
            cs=cats.get(int(m),[])
            w=1/math.log2(pos+2)
            for c in cs: self.exp[c,int(groups[pos])] += w/len(cs)
    def deficits(self,ids,groups,cats):
        totals=defaultdict(lambda:[0.,0.])
        for (c,g),o in self.opp.items():
            totals[c][0]+=self.exp[c,g]; totals[c][1]+=o
        ref={c:e/(o+EPS) if o>0 else 0. for c,(e,o) in totals.items()}
        out={}
        cells={(c,int(g)) for m,g in zip(ids,groups) for c in cats.get(int(m),[])}
        for c,g in cells:
            o=self.opp[c,g]; r=ref.get(c,0.)
            rate=self.exp[c,g]/(o+EPS) if o>EPS else 0.
            out[c,g]=max(0.,r-rate)/(r+EPS) if o>EPS and r>EPS else 0.
        return out
    def clone(self):
        x=Ledger(); x.exp.update(self.exp); x.opp.update(self.opp); return x

def rank_one(cand,item2group,cats,ledger,lam,gate):
    ids,scores,kg=cand["ids"],cand["scores"],cand["kg"]
    groups=np.asarray([item2group[int(m)] for m in ids],int)
    deficits=ledger.deficits(ids,groups,cats)
    bonus=np.zeros(len(ids)); categorized=eligible=0
    for j,m in enumerate(ids):
        cs=cats.get(int(m),[])
        if not cs: continue
        categorized+=1
        if gate and kg[j]<=0: continue
        eligible+=1
        bonus[j]=sum(deficits.get((c,int(groups[j])),0.) for c in cs)/len(cs)
    if bonus.max(initial=0)>0: bonus/=bonus.max()
    scale=float(np.median(np.abs(scores)))
    if scale<=EPS: scale=float(np.max(np.abs(scores))) if len(scores) else 1.
    order=np.lexsort((ids,-(scores+lam*scale*bonus)))
    return ids[order],groups[order],eligible,categorized,int(np.sum(order[:TOP_K]!=np.arange(min(TOP_K,len(order)))))

def metrics(ids,target):
    p=np.flatnonzero(ids[:TOP_K]==int(target))
    if not len(p): return 0,0.,None
    rank=int(p[0])+1
    return 1,1/math.log2(rank+1),rank

def disparity(exp,opp):
    cats=sorted({c for c,g in opp if opp[c,g]>EPS})
    vals=[]; detail=[]
    for c in cats:
        cells=[g for g in range(5) if opp[c,g]>EPS]
        if len(cells)<2: continue
        rates=np.array([exp[c,g]/(opp[c,g]+EPS) for g in cells])
        logs=np.log(np.maximum(rates,EPS)); mad=float(np.mean(np.abs(logs-logs.mean())))
        vals.append(mad)
        for g,r in zip(cells,rates):
            detail.append({"category":c,"popularity_group":g,"exposure":exp[c,g],
                           "opportunity_mass":opp[c,g],"exposure_per_opportunity":float(r),
                           "category_log_rate_mad":mad})
    return (float(np.mean(vals)) if vals else float("nan")),detail

def evaluate_fairness(frame,event_ids,data):
    exp,opp=defaultdict(float),defaultdict(float)
    for r in frame.sort_values("event_id").itertuples(index=False):
        ids=json.loads(r.top10_movie_ids); gs=json.loads(r.top10_group_ids)
        for pos,m in enumerate(ids):
            cs=data["cats"].get(int(m),[])
            for c in cs: exp[c,int(gs[pos])]+=1/math.log2(pos+2)/len(cs)
    for eid in event_ids:
        cand=data["cand"][eid]; w=Ledger.opp_weights(len(cand["ids"]))
        for j,m in enumerate(cand["ids"]):
            cs=data["cats"].get(int(m),[])
            for c in cs: opp[c,data["groups"][int(m)]]+=float(w[j])/len(cs)
    val,detail=disparity(exp,opp)
    return val,detail

def run_policy(data,event_ids,start_ledger,lam,gate):
    ledger=start_ledger.clone(); rows=[]
    for eid in sorted(event_ids):
        ev=data["events"][eid]; cand=data["cand"][eid]
        ids,gs,eligible,categorized,promoted=rank_one(
            cand,data["groups"],data["cats"],ledger,lam,gate)
        hit,ndcg,rank=metrics(ids,ev["target"])
        rows.append({"event_id":eid,"conv_id":ev["conv"],"hit10":hit,"ndcg10":ndcg,
                     "target_rank_top500":rank,"eligible_categorized_candidates":eligible,
                     "categorized_candidates":categorized,"top10_promoted_positions":promoted,
                     "top10_movie_ids":json.dumps(list(map(int,ids[:TOP_K]))),
                     "top10_group_ids":json.dumps(list(map(int,gs[:TOP_K])))})
        ledger.update(ids,data["cats"],gs)
    return pd.DataFrame(rows),ledger

def baseline_frame(data,event_ids):
    rows=[]
    for eid in sorted(event_ids):
        ev,c=data["events"][eid],data["cand"][eid]
        hit,ndcg,rank=metrics(c["ids"],ev["target"])
        ncat=sum(bool(data["cats"].get(int(m),[])) for m in c["ids"])
        rows.append({"event_id":eid,"conv_id":ev["conv"],"hit10":hit,"ndcg10":ndcg,
          "target_rank_top500":rank,"eligible_categorized_candidates":ncat,
          "categorized_candidates":ncat,"top10_promoted_positions":0,
          "top10_movie_ids":json.dumps(list(map(int,c["ids"][:TOP_K]))),
          "top10_group_ids":json.dumps([int(data["groups"][int(m)]) for m in c["ids"][:TOP_K]])})
    return pd.DataFrame(rows)

def summarize(frame):
    return {"events":len(frame),"conversations":int(frame.conv_id.nunique()),
            "HR@10":float(frame.hit10.mean()),"NDCG@10":float(frame.ndcg10.mean()),
            "mean_eligible_categorized_candidates":float(frame.eligible_categorized_candidates.mean()),
            "mean_top10_promoted_positions":float(frame.top10_promoted_positions.mean())}

def contribution_vectors(frame,event_ids,data,cat_index,ncat):
    expmap={}; oppmap={}
    by=frame.set_index("event_id")
    for eid in event_ids:
        r=by.loc[eid]; ev=np.zeros(ncat*5); ov=np.zeros(ncat*5)
        ids=json.loads(r.top10_movie_ids); gs=json.loads(r.top10_group_ids)
        for pos,(m,g) in enumerate(zip(ids,gs)):
            cs=data["cats"].get(int(m),[])
            for c in cs: ev[cat_index[c]*5+int(g)]+=1/math.log2(pos+2)/len(cs)
        cand=data["cand"][eid]; w=Ledger.opp_weights(len(cand["ids"]))
        for j,m in enumerate(cand["ids"]):
            cs=data["cats"].get(int(m),[]); g=data["groups"][int(m)]
            for c in cs: ov[cat_index[c]*5+g]+=float(w[j])/len(cs)
        expmap[eid]=ev; oppmap[eid]=ov
    return expmap,oppmap

def disparity_vectors(e,o,ncat):
    e=e.reshape(ncat,5); o=o.reshape(ncat,5); vals=[]
    for i in range(ncat):
        mask=o[i]>EPS
        if mask.sum()<2: continue
        z=np.log(np.maximum(e[i,mask]/(o[i,mask]+EPS),EPS))
        vals.append(float(np.mean(np.abs(z-z.mean()))))
    return float(np.mean(vals)) if vals else float("nan")

def bootstrap_compare(frames,contrib,event_conv,held_conv_ids,categories,nboot):
    methods=list(frames); ncat=len(categories)
    ev_by_conv={c:[] for c in held_conv_ids}
    for eid,c in event_conv.items():
        if c in ev_by_conv: ev_by_conv[c].append(eid)
    convs=np.asarray(sorted(held_conv_ids),dtype=np.int64)
    rng=np.random.default_rng(BOOT_SEED)
    samples=rng.choice(convs,size=(nboot,len(convs)),replace=True)
    indexed={m:frames[m].set_index("event_id") for m in methods}
    distributions={m:{"fair":[],"hr":[],"ndcg":[]} for m in methods if m!="B0_KBRD"}
    for sample in samples:
        sampled=[]
        for c in sample: sampled.extend(ev_by_conv[int(c)])
        util={}; fair={}
        for m in methods:
            v=indexed[m].loc[sampled][["hit10","ndcg10"]].to_numpy(float)
            util[m]=v.mean(axis=0)
            ex=np.zeros(ncat*5); op=np.zeros(ncat*5)
            em,om=contrib[m]
            for eid in sampled: ex+=em[eid]; op+=om[eid]
            fair[m]=disparity_vectors(ex,op,ncat)
        for m in distributions:
            distributions[m]["fair"].append(fair[m]-fair["B0_KBRD"])
            distributions[m]["hr"].append(util[m][0]-util["B0_KBRD"][0])
            distributions[m]["ndcg"].append(util[m][1]-util["B0_KBRD"][1])
    rows=[]
    point={}
    for m in methods:
        ix=frames[m]
        point[m]=(float(ix.hit10.mean()),float(ix.ndcg10.mean()))
    # Point fairness values are calculated from full held-out vectors.
    full_fair={}
    for m in methods:
        ex=np.zeros(ncat*5); op=np.zeros(ncat*5); em,om=contrib[m]
        for eid in em: ex+=em[eid]; op+=om[eid]
        full_fair[m]=disparity_vectors(ex,op,ncat)
    for m,d in distributions.items():
        def ci(k):
            a=np.asarray(d[k]); return float(np.quantile(a,.025)),float(np.quantile(a,.975))
        fl,fh=ci("fair"); hl,hh=ci("hr"); nl,nh=ci("ndcg")
        rows.append({"method":m,"delta_fairness_method_minus_B0":full_fair[m]-full_fair["B0_KBRD"],
          "fairness_delta_ci95_low":fl,"fairness_delta_ci95_high":fh,
          "bootstrap_probability_fairness_improves":float(np.mean(np.asarray(d["fair"])<0)),
          "delta_HR@10":point[m][0]-point["B0_KBRD"][0],"delta_HR@10_ci95_low":hl,
          "delta_HR@10_ci95_high":hh,"delta_NDCG@10":point[m][1]-point["B0_KBRD"][1],
          "delta_NDCG@10_ci95_low":nl,"delta_NDCG@10_ci95_high":nh,
          "bootstrap_replicates":nboot,"conversation_clusters":len(convs)})
    return pd.DataFrame(rows),full_fair

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kbrd",type=Path,default=Path("results_corrected/experiment_22_04B/kbrd_validation_top500.csv"))
    ap.add_argument("--manifest",type=Path,default=Path("results_corrected/experiment_22_04B/kbrd_validation_event_manifest.csv"))
    ap.add_argument("--kg",type=Path,default=Path("results_corrected/adaptive_m2_validation_export22_candidates.csv"))
    ap.add_argument("--catalog-pickle",type=Path,default=Path("catalog_audit_corrected.pkl"))
    ap.add_argument("--category-profiles",type=Path,default=Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"))
    ap.add_argument("--out",type=Path,default=Path("results_corrected/experiment_22_09"))
    ap.add_argument("--bootstrap",type=int,default=10000)
    ap.add_argument("--lambdas",type=str,default=",".join(map(str,LAMBDAS)))
    a=ap.parse_args()
    if a.bootstrap<100: raise ValueError("Use at least 100 bootstrap replicates; final run should use 10000.")
    lambdas=sorted(set(float(x) for x in a.lambdas.split(",")))
    if not lambdas or min(lambdas)<0: raise ValueError("Invalid lambda grid.")
    a.out.mkdir(parents=True,exist_ok=True)

    print("Loading frozen KBRD, KG, catalog, and category inputs...")
    meta,groups,catalog,cats,profiles=load_metadata(a.catalog_pickle,a.category_profiles)
    k=pd.read_csv(a.kbrd); mf=pd.read_csv(a.manifest)
    kg=pd.read_csv(a.kg,usecols=["event_id","method","candidate_movie_ids","candidate_kg_ranks"])
    kg=kg[kg.method=="A1_frequency_3"].copy()
    for f in (k,mf,kg): f["event_id"]=f.event_id.astype(int)
    for f,n in ((k,"KBRD"),(mf,"manifest"),(kg,"KG")):
        if f.event_id.duplicated().any(): raise ValueError(f"Duplicate {n} event IDs.")
    if set(k.event_id)!=set(mf.event_id) or set(k.event_id)!=set(kg.event_id):
        raise ValueError("KBRD, manifest, and KG event IDs do not match.")
    mf=mf[["event_id","conv_id","target_movie_id"]]
    k=k.merge(mf,on="event_id",how="inner",validate="one_to_one",suffixes=("","_manifest"))
    if "conv_id_manifest" in k and not np.array_equal(k.conv_id,k.conv_id_manifest):
        raise ValueError("KBRD/manifest conversation IDs mismatch.")
    if "target_movie_id_manifest" in k and not np.array_equal(k.target_movie_id,k.target_movie_id_manifest):
        raise ValueError("KBRD/manifest target IDs mismatch.")

    kgmap={}
    for r in kg.itertuples(index=False):
        mids=np.asarray(as_list(r.candidate_movie_ids,"KG candidate IDs"),dtype=np.int64)
        ranks=np.asarray(as_list(r.candidate_kg_ranks,"KG ranks"),dtype=np.int64)
        if len(mids)!=len(ranks) or len(np.unique(mids))!=len(mids) or np.any(ranks<0):
            raise ValueError(f"Invalid KG candidate/rank arrays at event {r.event_id}.")
        kgmap[int(r.event_id)]={int(m):int(v) for m,v in zip(mids,ranks)}

    events={}; cand={}
    for r in k.sort_values("event_id").itertuples(index=False):
        eid=int(r.event_id); ids=np.asarray(as_list(r.top_movie_ids,"KBRD IDs"),dtype=np.int64)
        scores=np.asarray(as_list(r.top_scores,"KBRD scores"),dtype=float)
        if len(ids)!=CANDIDATE_K or len(scores)!=CANDIDATE_K or len(np.unique(ids))!=CANDIDATE_K:
            raise ValueError(f"Invalid KBRD Top-500 at event {eid}.")
        if not np.isfinite(scores).all() or not set(map(int,ids)).issubset(catalog):
            raise ValueError(f"Invalid scores/catalog IDs at event {eid}.")
        kr=kgmap[eid]; transferred=np.asarray([kr.get(int(m),0) for m in ids],dtype=np.int64)
        events[eid]={"conv":int(r.conv_id),"target":int(r.target_movie_id)}
        cand[eid]={"ids":ids,"scores":scores,"kg":transferred}
    if len(events)!=3010 or len(set(v["conv"] for v in events.values()))!=883:
        raise ValueError("Expected 3010 events and 883 conversations.")
    data={"events":events,"cand":cand,"groups":groups,"cats":cats}

    eids=sorted(events); event_conv={e:events[e]["conv"] for e in eids}
    warm,tune,held=split_conversations(event_conv.values())
    byconv=defaultdict(list)
    for e,c in event_conv.items(): byconv[c].append(e)
    warm_e=sorted(e for c in warm for e in byconv[c])
    tune_e=sorted(e for c in tune for e in byconv[c])
    held_e=sorted(e for c in held for e in byconv[c])
    if set(warm_e)|set(tune_e)|set(held_e)!=set(eids) or (warm&tune) or (warm&held) or (tune&held):
        raise RuntimeError("Conversation split integrity failure.")
    print(f"Split events: warm={len(warm_e)}, tune={len(tune_e)}, heldout={len(held_e)}")

    # KBRD-only warm-up ledger; no KG-backbone recommendations enter this history.
    warm_ledger=Ledger()
    for eid in warm_e:
        c=cand[eid]; gs=np.asarray([groups[int(m)] for m in c["ids"]],int)
        warm_ledger.update(c["ids"],cats,gs)

    b0_tune=baseline_frame(data,tune_e)
    baseline_ndcg=float(b0_tune.ndcg10.mean())
    tuning=[]
    selected={}
    for name,gate in (("B1_KBRD_category_only",False),("B2_KBRD_KG_gated",True)):
        feasible=[]
        for lam in lambdas:
            trial,_=run_policy(data,tune_e,warm_ledger,lam,gate)
            fair,_=evaluate_fairness(trial,tune_e,data)
            nd=float(trial.ndcg10.mean())
            row={"method":name,"lambda":lam,"HR@10":float(trial.hit10.mean()),
                 "NDCG@10":nd,"category_group_log_rate_MAD":fair,
                 "retention_vs_B0":nd/(baseline_ndcg+EPS),
                 "meets_95pct_utility":bool(nd>=.95*baseline_ndcg)}
            tuning.append(row)
            if row["meets_95pct_utility"] and np.isfinite(fair): feasible.append(row)
        if feasible:
            selected[name]=float(sorted(feasible,key=lambda x:(x["category_group_log_rate_MAD"],-x["NDCG@10"],x["lambda"]))[0]["lambda"])
        else:
            selected[name]=0.0
        print(f"{name}: lambda={selected[name]} (fallback 0 if no feasible setting)")
    pd.DataFrame(tuning).to_csv(a.out/"lambda_selection.csv",index=False)

    frames={"B0_KBRD":baseline_frame(data,held_e)}
    for name,gate in (("B1_KBRD_category_only",False),("B2_KBRD_KG_gated",True)):
        _,ledger_after_tune=run_policy(data,tune_e,warm_ledger,selected[name],gate)
        frames[name],_=run_policy(data,held_e,ledger_after_tune,selected[name],gate)

    point_rows=[]; fair_rows=[]; details={}
    for name,frame in frames.items():
        s=summarize(frame); s["method"]=name
        s["selected_lambda"]=0.0 if name=="B0_KBRD" else selected[name]
        point_rows.append(s)
        f,detail=evaluate_fairness(frame,held_e,data)
        fair_rows.append({"method":name,"category_group_log_rate_MAD":f,
                          "categories_with_supported_groups":len({x["category"] for x in detail})})
        details[name]=detail
    pd.DataFrame(point_rows).to_csv(a.out/"point_estimates.csv",index=False)
    pd.DataFrame(fair_rows).to_csv(a.out/"category_fairness_results.csv",index=False)
    for name,detail in details.items():
        pd.DataFrame(detail).to_csv(a.out/f"fairness_detail_{name}.csv",index=False)
    pd.concat([v.assign(method=k) for k,v in frames.items()],ignore_index=True).to_csv(
        a.out/"event_level_results.csv",index=False)

    category_names=sorted({c for v in cats.values() for c in v})
    cat_index={c:i for i,c in enumerate(category_names)}
    contrib={name:contribution_vectors(frame,held_e,data,cat_index,len(category_names))
             for name,frame in frames.items()}
    held_conv_ids=set(event_conv[e] for e in held_e)
    boot,full_fair=bootstrap_compare(frames,contrib,event_conv,held_conv_ids,
                                     category_names,a.bootstrap)
    boot.to_csv(a.out/"paired_bootstrap_results.csv",index=False)

    kg_counts=np.asarray([int(np.sum(cand[e]["kg"]>0)) for e in eids])
    checks={
      "experiment":"22.09_KBRD_cross_backbone_validation","validation_only":True,
      "test_split_evaluated":False,"events":len(eids),"conversations":len(set(event_conv.values())),
      "warm_events":len(warm_e),"tuning_events":len(tune_e),"heldout_events":len(held_e),
      "warm_conversations":len(warm),"tuning_conversations":len(tune),"heldout_conversations":len(held),
      "split_seed":SEED,"lambda_grid":lambdas,"selected_lambda":selected,
      "utility_constraint":"NDCG@10 >= 95% of original KBRD on tuning conversations",
      "candidate_pool":"KBRD Top-500","kg_transfer_key":"(event_id,movie_id)",
      "missing_kg_evidence_default":0,"kg_support_definition":"KG rank > 0; retrieval support, not semantic relevance",
      "kg_supported_candidate_fraction":float(np.mean(np.concatenate([cand[e]["kg"]>0 for e in eids]))),
      "events_with_kg_supported_candidates":int(np.sum(kg_counts>0)),
      "events_without_kg_supported_candidates":int(np.sum(kg_counts==0)),
      "bootstrap_replicates":a.bootstrap,"bootstrap_seed":BOOT_SEED,
      "bootstrap_sampling_unit":"held-out conversation_id","same_resamples_for_all_methods":True,
      "methods":list(frames),"test_data_loaded":False
    }
    with open(a.out/"validation_checks.json","w",encoding="utf-8") as f: json.dump(checks,f,indent=2)
    manifest={**checks,"inputs":{"kbrd":str(a.kbrd),"manifest":str(a.manifest),"kg":str(a.kg),
      "catalog_pickle":str(a.catalog_pickle),"category_profiles":str(a.category_profiles)},
      "outputs":["point_estimates.csv","event_level_results.csv","category_fairness_results.csv",
      "paired_bootstrap_results.csv","lambda_selection.csv","validation_checks.json"],
      "limitations":["KG rank is a retrieval-support signal, not semantic relevance.",
      "External category-label uncertainty is not included in bootstrap intervals.",
      "This is validation-only evidence, not independent test-set evaluation."]}
    with open(a.out/"experiment_manifest.json","w",encoding="utf-8") as f: json.dump(manifest,f,indent=2)
    print("\nEXPERIMENT 22.9 COMPLETE")
    print("\nHeld-out point estimates:\n",pd.DataFrame(point_rows).to_string(index=False))
    print("\nPaired bootstrap vs original KBRD:\n",boot.to_string(index=False))
    print("\nOutputs:",a.out.resolve())
    print("Test split evaluated: NO")

if __name__=="__main__": main()
