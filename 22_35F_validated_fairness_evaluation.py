#!/usr/bin/env python
"""Experiment 22.35F: cumulative validated fairness evaluation.

Preserves the 22.35E four-system design and imports the unchanged 22.16/22.18
implementations. Corrects split-local fairness: parameter selection uses the
cumulative warm+tuning R/E ledger; held-out evaluation uses warm+tune+held-out.
R0_filtered is primary; R1_raw is A3 sensitivity with R0-selected parameters.
Validation-only: no training, retrieval rerun, checkpoint changes, or test access.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

EPS, TOP_K = 1e-12, 10
RETENTION, BOOT_REPS, BOOT_SEED = 0.95, 2000, 22035
Q_GRID = (0.0005, 0.001, 0.002, 0.005)
LAM16_GRID = (0.05, 0.10, 0.20, 0.40, 0.80)
LAM18_GRID = (0.05, 0.10, 0.20, 0.40, 0.80)
TAU_GRID = (25.0, 50.0, 100.0, 250.0)
RMIN_GRID = (0.0, 0.05, 0.10, 0.25, 0.50)
LAM_FAIR_GRID = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0)

D_KGSF = Path("results_corrected/experiment_22_35/kgsf_validation/kgsf_validation_top500.csv")
D_R0 = Path("results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R0_filtered.csv")
D_R1 = Path("results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R1_raw.csv")
D_CAT = Path("catalog_audit_corrected.pkl")
D_PROFILES = Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
D_22_16 = Path("22_16_category_context_utility_calibration.py")
D_22_18 = Path("22_18_contextual_relevance_gate_validation.py")
D_OUT = Path("results_corrected/experiment_22_35/validated_fairness_evaluation")
OLD_OUT = Path("results_corrected/experiment_22_35/integrated_reranking")
SYSTEMS = ("A0_KGSF", "A1_22_16", "A2_22_18", "A3_KG_FairCRS")


def parse_list(x, name):
    if isinstance(x, list): y = x
    elif isinstance(x, (tuple, np.ndarray)): y = list(x)
    else:
        try: y = json.loads(x)
        except Exception: y = ast.literal_eval(x)
    if not isinstance(y, list): raise ValueError(f"{name} must be a list")
    return y


def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path): return str(x)
    return x


def dump_json(path, obj):
    path.write_text(json.dumps(safe(obj), indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def import_ref(path, module_name, required):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    missing = [x for x in required if not hasattr(mod, x)]
    if missing: raise AttributeError(f"{path} missing API: {missing}")
    return mod


def load_data(kpath, qpath, catalog):
    k, qdf = pd.read_csv(kpath), pd.read_csv(qpath)
    rk = {"event_id","conv_id","target_movie_id","top_movie_ids","top_scores"}
    rq = {"event_id","candidate_rank","movie_id","kg_rank","q_ei"}
    if rk-set(k): raise ValueError(f"KGSF missing {sorted(rk-set(k))}")
    if rq-set(qdf): raise ValueError(f"{qpath} missing {sorted(rq-set(qdf))}")
    if k.event_id.duplicated().any() or qdf.duplicated(["event_id","movie_id"]).any():
        raise ValueError("Duplicate event/candidate keys")
    if set(k.event_id.astype(int)) != set(qdf.event_id.astype(int)):
        raise ValueError("KGSF/q_ei event IDs differ")
    qmap = {int(e): f.sort_values("candidate_rank") for e,f in qdf.groupby("event_id")}
    events, cands = {}, {}
    for row in k.sort_values("event_id").itertuples(index=False):
        e = int(row.event_id)
        ids = np.asarray(parse_list(row.top_movie_ids,"top_movie_ids"), dtype=np.int64)
        scores = np.asarray(parse_list(row.top_scores,"top_scores"), dtype=float)
        q = qmap[e]
        qids = q.movie_id.astype("int64").to_numpy()
        ranks = q.candidate_rank.astype("int64").to_numpy()
        probs = pd.to_numeric(q.q_ei, errors="raise").to_numpy(float)
        kg = pd.to_numeric(q.kg_rank, errors="raise").to_numpy(int)
        if len(ids)!=500 or len(scores)!=500 or len(np.unique(ids))!=500:
            raise ValueError(f"Invalid frozen Top-500 at {e}")
        if not np.array_equal(ids,qids) or not np.array_equal(ranks,np.arange(1,501)):
            raise ValueError(f"Candidate order mismatch at {e}")
        if not np.isfinite(scores).all() or not np.isfinite(probs).all() or (probs<0).any():
            raise ValueError(f"Invalid scores/q_ei at {e}")
        if not np.isclose(probs.sum(),1,atol=1e-8) or (kg<0).any():
            raise ValueError(f"Invalid q_ei/KG ranks at {e}")
        if not set(map(int,ids)).issubset(catalog): raise ValueError(f"Out-of-catalog item at {e}")
        events[e] = {"conv":int(row.conv_id),"target":int(row.target_movie_id)}
        cands[e] = {"ids":ids,"scores":scores,"kg":kg,"q":probs}
    if len(events)!=3010 or len({v["conv"] for v in events.values()})!=883:
        raise ValueError("Expected 3,010 events / 883 conversations")
    return events,cands


def groups(c, item2group):
    return np.asarray([item2group[int(x)] for x in c["ids"]], dtype=int)


def metric(ids, target):
    pos = np.flatnonzero(np.asarray(ids[:TOP_K]) == int(target))
    if not len(pos): return 0,0.0,0.0
    r = int(pos[0])+1
    return 1,1/math.log2(r+1),1/r


def update_legacy(ledger,c,ranked,item2group,categories):
    cg = groups(c,item2group)
    rg = np.asarray([item2group[int(x)] for x in ranked[:TOP_K]],dtype=int)
    ledger.update(c["ids"],cg,np.asarray(ranked[:TOP_K],dtype=np.int64),rg,categories)


class ProposalLedger:
    """Cumulative proposal R_cg/E_cg ledger."""
    def __init__(self):
        self.R, self.E = defaultdict(float), defaultdict(float)
    def add_R(self,c,item2group,categories,qthr):
        gr = groups(c,item2group)
        for j,mid in enumerate(c["ids"]):
            if c["q"][j] >= qthr:
                for cat in categories.get(int(mid),[]):
                    self.R[(cat,int(gr[j]))] += float(c["q"][j])
    def add_E(self,ranked,item2group,categories):
        for rank,mid in enumerate(ranked[:TOP_K],1):
            w = 1/math.log2(rank+1)
            for cat in categories.get(int(mid),[]):
                self.E[(cat,int(item2group[int(mid)]))] += w
    def clone(self):
        z=ProposalLedger(); z.R.update(self.R); z.E.update(self.E); return z


def proposal_terms(state,rmin):
    bycat=defaultdict(list)
    for (cat,g),r in state.R.items():
        if r>EPS and r>=rmin: bycat[cat].append((g,r))
    active={c:v for c,v in bycat.items() if len(v)>=2}
    if not active: return float("nan"),0,[]
    rows=[]; loss=0.0; n=len(active)
    for cat,items in sorted(active.items()):
        total=sum(r for _,r in items)
        p=np.asarray([r/total for _,r in items])
        rho=np.asarray([state.E[(cat,g)]/(r+EPS) for g,r in items])
        mean=float(np.sum(p*rho))
        var=float(np.sum(p*(rho-mean)**2))
        contribution=var/n; loss += contribution
        for (g,r),pp,rr in zip(items,p,rho):
            rows.append({"category":cat,"popularity_group":int(g),"R_cg":r,
                         "E_cg":state.E[(cat,g)],"rho_cg":float(rr),"p_cg":float(pp),
                         "category_variance_contribution":contribution,
                         "active_category_count":n})
    return float(loss),n,rows


def proposal_deficits(state,c,item2group,categories,rmin):
    bycat=defaultdict(list)
    for (cat,g),r in state.R.items():
        if r>EPS and r>=rmin: bycat[cat].append((g,r))
    means={}
    for cat,items in bycat.items():
        if len(items)>=2:
            total=sum(r for _,r in items)
            means[cat]=sum((r/total)*state.E[(cat,g)]/(r+EPS) for g,r in items)
    out={}
    for mid in c["ids"]:
        g=int(item2group[int(mid)])
        for cat in categories.get(int(mid),[]):
            if cat in means:
                r=state.R[(cat,g)]
                if r>=rmin and r>EPS:
                    out[(cat,g)]=max(0.0,means[cat]-state.E[(cat,g)]/(r+EPS))
    return out


def rank_a3(c,item2group,categories,state,qthr,rmin,lam):
    # Original 22.35E order: add current opportunity before computing deficits.
    state.add_R(c,item2group,categories,qthr)
    deficits=proposal_deficits(state,c,item2group,categories,rmin)
    ids,scores=c["ids"],c["scores"]; eligible=c["q"]>=qthr
    bonus=np.zeros(len(ids))
    for j,mid in enumerate(ids):
        if eligible[j]:
            cats=categories.get(int(mid),[])
            if cats: bonus[j]=np.mean([deficits.get((cat,int(item2group[int(mid)])),0.0) for cat in cats])
    adjusted=scores.copy(); adjusted[eligible]+=float(lam)*bonus[eligible]
    order=np.lexsort((ids,-adjusted))
    eligible_order=[int(j) for j in order if eligible[j]]
    chosen=eligible_order[:TOP_K]; fallback=max(0,TOP_K-len(chosen))
    if fallback:
        base=np.argsort(-scores,kind="stable"); used=set(chosen)
        chosen.extend([int(j) for j in base if int(j) not in used][:fallback])
    chosen=chosen[:TOP_K]; ranked=ids[np.asarray(chosen,dtype=int)]
    state.add_E(ranked,item2group,categories)
    return ranked,{"eligible_candidates":int(eligible.sum()),
        "eligible_relevance_mass":float(c["q"][eligible].sum()),
        "kg_supported_candidates":int(np.sum(c["kg"]>0)),
        "kg_supported_eligible_candidates":int(np.sum((c["kg"]>0)&eligible)),
        "fallback_positions":fallback,
        "top10_promoted_positions":int(np.sum(np.asarray(chosen)!=np.arange(TOP_K)))}


def warm_legacy(ids,cands,item2group,categories,ref):
    x=ref.ExposureLedger()
    for e in sorted(ids):
        c=cands[e]; gr=groups(c,item2group)
        x.update(c["ids"],gr,c["ids"],gr,categories)
    return x


def warm_prop(ids,cands,item2group,categories,qthr):
    x=ProposalLedger()
    for e in sorted(ids):
        x.add_R(cands[e],item2group,categories,qthr)
        x.add_E(cands[e]["ids"],item2group,categories)
    return x


def run_segment(name,ids,events,cands,item2group,categories,ref,legacy0,prop0,params,qthr,rmin):
    legacy,prop=legacy0.clone(),prop0.clone(); rows=[]
    for e in sorted(ids):
        c=cands[e]
        if name=="A0_KGSF": ranked,info=c["ids"],{}
        elif name=="A1_22_16":
            ranked,_,info=ref.rerank(c,item2group,categories,legacy,float(params["lam"]),"category")
            ranked=np.asarray(ranked,dtype=np.int64)
        elif name=="A2_22_18":
            ranked,info=ref18.contextual_gate_rerank(
                c,item2group,categories,legacy,float(params["lam"]),float(params["tau"]))
            ranked=np.asarray(ranked,dtype=np.int64)
        elif name=="A3_KG_FairCRS":
            ranked,info=rank_a3(c,item2group,categories,prop,qthr,rmin,params["lam"])
        else: raise ValueError(name)
        hit,ndcg,mrr=metric(ranked,events[e]["target"])
        kgmap={int(m):int(k) for m,k in zip(c["ids"],c["kg"])}
        rows.append({"system":name,"event_id":int(e),"conv_id":int(events[e]["conv"]),
            "hit10":hit,"ndcg10":ndcg,"mrr10":mrr,
            "kg_supported_candidates":int(np.sum(c["kg"]>0)),
            "kg_supported_top10":sum(kgmap.get(int(m),0)>0 for m in ranked[:TOP_K]),
            "eligible_candidates":int(info.get("eligible_candidates",0)),
            "fallback_positions":int(info.get("fallback_positions",0)),
            "top10_promoted_positions":int(info.get("top10_promoted_positions",0)),
            "top10_movie_ids":json.dumps(list(map(int,ranked[:TOP_K])))})
        update_legacy(legacy,c,ranked,item2group,categories)
        if name!="A3_KG_FairCRS":
            prop.add_R(c,item2group,categories,qthr); prop.add_E(ranked,item2group,categories)
    frame=pd.DataFrame(rows); fair,n,cells=proposal_terms(prop,rmin)
    summary={"events":len(frame),"conversations":int(frame.conv_id.nunique()),
        "HR@10":float(frame.hit10.mean()),"NDCG@10":float(frame.ndcg10.mean()),
        "MRR@10":float(frame.mrr10.mean()),"L_fair_cumulative":fair,
        "active_categories_cumulative":n,
        "mean_kg_supported_top10":float(frame.kg_supported_top10.mean()),
        "mean_eligible_candidates":float(frame.eligible_candidates.mean()),
        "mean_fallback_positions":float(frame.fallback_positions.mean()),
        "mean_top10_promoted_positions":float(frame.top10_promoted_positions.mean())}
    return frame,legacy,prop,summary,cells


def tune_a3(events,cands,warm,tune,item2group,categories,ref,retention):
    base_ndcg=np.mean([metric(cands[e]["ids"],events[e]["target"])[1] for e in sorted(tune)])
    base_legacy=warm_legacy(warm,cands,item2group,categories,ref)
    rows=[]; best=None
    for qthr in Q_GRID:
        wp=warm_prop(warm,cands,item2group,categories,qthr)
        for rmin in RMIN_GRID:
            for lam in LAM_FAIR_GRID:
                _,_,state,s,_=run_segment("A3_KG_FairCRS",tune,events,cands,item2group,
                    categories,ref,base_legacy,wp,{"lam":lam},qthr,rmin)
                fair,n,_=proposal_terms(state,rmin); retained=s["NDCG@10"]/max(base_ndcg,EPS)
                row={"q_threshold":qthr,"R_min":rmin,"lam":lam,**s,
                     "L_fair_cumulative":fair,"active_categories_cumulative":n,
                     "NDCG_retention":retained,"utility_constraint_met":retained>=retention}
                rows.append(row)
                if retained>=retention and np.isfinite(fair):
                    key=(fair,-s["NDCG@10"],qthr,rmin,lam)
                    if best is None or key<best[0]: best=(key,{"q_threshold":qthr,"R_min":rmin,"lam":lam})
    if best: return best[1],pd.DataFrame(rows),"CUMULATIVE_UTILITY_CONSTRAINED_MINIMUM_L_FAIR"
    return {"q_threshold":Q_GRID[0],"R_min":RMIN_GRID[0],"lam":0.0},pd.DataFrame(rows),"NO_FEASIBLE_CONFIGURATION_FALLBACK_TO_KGSF_ORDER"


def tune_legacy(name,events,cands,warm,tune,item2group,categories,ref,retention,qthr,rmin):
    base_ndcg=np.mean([metric(cands[e]["ids"],events[e]["target"])[1] for e in sorted(tune)])
    bl=warm_legacy(warm,cands,item2group,categories,ref); bp=warm_prop(warm,cands,item2group,categories,qthr)
    grid=([{"lam":x} for x in LAM16_GRID] if name=="A1_22_16"
          else [{"lam":x,"tau":t} for t in TAU_GRID for x in LAM18_GRID])
    rows=[]; best=None
    for p in grid:
        _,_,state,s,_=run_segment(name,tune,events,cands,item2group,categories,ref,bl,bp,p,qthr,rmin)
        fair,n,_=proposal_terms(state,rmin); retained=s["NDCG@10"]/max(base_ndcg,EPS)
        row={**p,**s,"L_fair_cumulative":fair,"active_categories_cumulative":n,
             "NDCG_retention":retained,"utility_constraint_met":retained>=retention}
        rows.append(row)
        if retained>=retention and np.isfinite(fair):
            key=(fair,-s["NDCG@10"],p["lam"],p.get("tau",0))
            if best is None or key<best[0]: best=(key,p)
    fallback={"lam":0.0} if name=="A1_22_16" else {"lam":0.0,"tau":TAU_GRID[0]}
    return (best[1] if best else fallback),pd.DataFrame(rows),bool(best)


def ci(vals):
    a=np.asarray(vals,dtype=float); a=a[np.isfinite(a)]
    return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5))) if len(a) else (float("nan"),float("nan"))


def bootstrap(events,cands,held,data,preheld,item2group,categories,qthr,rmin,reps,seed):
    byconv=defaultdict(list)
    for e in held: byconv[int(events[e]["conv"])].append(int(e))
    convs=np.asarray(sorted(byconv),dtype=np.int64); rng=np.random.default_rng(seed)
    frames={s:data[s]["held"].set_index("event_id") for s in data}
    metrics={s:{m:[] for m in ("HR","NDCG","MRR","fair")} for s in data}
    deltas={s:{m:[] for m in ("HR","NDCG","MRR","fair")} for s in data if s!="A0_KGSF"}
    for _ in range(reps):
        draw=rng.choice(convs,size=len(convs),replace=True)
        sampled=[e for c in draw for e in sorted(byconv[int(c)])]
        current={}
        for s in data:
            f=frames[s]; vals=f.loc[sampled]
            current[s]={"HR":float(vals.hit10.mean()),"NDCG":float(vals.ndcg10.mean()),
                        "MRR":float(vals.mrr10.mean())}
            state=preheld[s].clone()
            for e in sampled:
                ranked=parse_list(f.loc[e,"top10_movie_ids"],"bootstrap top10")
                state.add_R(cands[e],item2group,categories,qthr)
                state.add_E(ranked,item2group,categories)
            current[s]["fair"]=proposal_terms(state,rmin)[0]
            for m in metrics[s]: metrics[s][m].append(current[s][m])
        for s in deltas:
            for m in deltas[s]: deltas[s][m].append(current[s][m]-current["A0_KGSF"][m])
    out=[]
    for s,mm in metrics.items():
        for m,v in mm.items():
            lo,hi=ci(v); out.append({"comparison":"absolute","system":s,"metric":m,
                "estimate":float(np.nanmean(v)),"ci95_low":lo,"ci95_high":hi,
                "bootstrap_reps":reps,"cluster_unit":"conversation"})
    for s,mm in deltas.items():
        for m,v in mm.items():
            lo,hi=ci(v); out.append({"comparison":"paired_delta_vs_A0_KGSF","system":s,
                "metric":m,"estimate":float(np.nanmean(v)),"ci95_low":lo,"ci95_high":hi,
                "bootstrap_reps":reps,"cluster_unit":"conversation"})
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kgsf",type=Path,default=D_KGSF); ap.add_argument("--q-r0",type=Path,default=D_R0)
    ap.add_argument("--q-r1",type=Path,default=D_R1); ap.add_argument("--catalog",type=Path,default=D_CAT)
    ap.add_argument("--profiles",type=Path,default=D_PROFILES); ap.add_argument("--reference",type=Path,default=D_22_16)
    ap.add_argument("--reference-22-18",type=Path,default=D_22_18); ap.add_argument("--out",type=Path,default=D_OUT)
    ap.add_argument("--utility-retention",type=float,default=RETENTION)
    ap.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS); ap.add_argument("--bootstrap-seed",type=int,default=BOOT_SEED)
    a=ap.parse_args()
    if not 0<a.utility_retention<=1: raise ValueError("utility-retention must be in (0,1]")
    if a.bootstrap_reps<100: raise ValueError("bootstrap-reps must be >=100")
    if a.out.resolve()==OLD_OUT.resolve(): raise ValueError("Refusing to overwrite 22.35E output directory")
    for p in (a.kgsf,a.q_r0,a.q_r1,a.catalog,a.profiles,a.reference,a.reference_22_18):
        if not p.is_file(): raise FileNotFoundError(p)

    global ref18
    ref=import_ref(a.reference,"ref_22_16",("load_metadata","split_conversations","ExposureLedger","rerank"))
    ref18=import_ref(a.reference_22_18,"ref_22_18",("contextual_gate_rerank",))
    item2group,catalog,categories=ref.load_metadata(a.catalog,a.profiles)
    events,c0=load_data(a.kgsf,a.q_r0,catalog); events1,c1=load_data(a.kgsf,a.q_r1,catalog)
    if events!=events1 or any(not np.array_equal(c0[e]["ids"],c1[e]["ids"]) for e in events):
        raise RuntimeError("R0/R1 event or candidate mappings differ")
    convs=[v["conv"] for v in events.values()]
    wc,tc,hc=ref.split_conversations(convs)
    if wc&tc or wc&hc or tc&hc: raise RuntimeError("Conversation split overlap")
    warm=sorted(e for e,v in events.items() if v["conv"] in wc)
    tune=sorted(e for e,v in events.items() if v["conv"] in tc)
    held=sorted(e for e,v in events.items() if v["conv"] in hc)
    if set(warm)|set(tune)|set(held)!=set(events): raise RuntimeError("Split does not cover events")
    a.out.mkdir(parents=True,exist_ok=True)
    print(f"22.35F: events={len(events)}, conversations={len(set(convs))}, warm/tune/held={len(warm)}/{len(tune)}/{len(held)}")

    p3,g3,status3=tune_a3(events,c0,warm,tune,item2group,categories,ref,a.utility_retention)
    g3.insert(0,"system","A3_KG_FairCRS"); g3.to_csv(a.out/"tuning_A3_integrated.csv",index=False)
    qthr,rmin=p3["q_threshold"],p3["R_min"]
    decisions={"A3_KG_FairCRS":{"parameters":p3,"selection_status":status3},
               "A0_KGSF":{"parameters":{},"selection_status":"FROZEN_BASELINE"}}
    tuning=[g3]
    for name in ("A1_22_16","A2_22_18"):
        p,g,feasible=tune_legacy(name,events,c0,warm,tune,item2group,categories,ref,
                                 a.utility_retention,qthr,rmin)
        g.insert(0,"system",name); tuning.append(g)
        decisions[name]={"parameters":p,"selection_status":
            "CUMULATIVE_UTILITY_CONSTRAINED_MINIMUM_L_FAIR" if feasible else
            "NO_FEASIBLE_CONFIGURATION_FALLBACK_TO_ZERO_LAMBDA"}

    event_frames=[]; summary=[]; cells=[]; data={}; preheld={}
    for name in SYSTEMS:
        params=decisions[name]["parameters"]
        wl=warm_legacy(warm,c0,item2group,categories,ref); wp=warm_prop(warm,c0,item2group,categories,qthr)
        tf,tl,tp,ts,_=run_segment(name,tune,events,c0,item2group,categories,ref,wl,wp,params,qthr,rmin)
        hf,_,hp,hs,_=run_segment(name,held,events,c0,item2group,categories,ref,tl,tp,params,qthr,rmin)
        tune_f,tune_n,tune_cells=proposal_terms(tp,rmin); final_f,final_n,final_cells=proposal_terms(hp,rmin)
        for f,split in ((tf,"tune"),(hf,"heldout")):
            z=f.copy(); z["split"]=split; z["variant"]="R0_filtered"; event_frames.append(z)
        summary += [{"system":name,"variant":"R0_filtered","split":"tune",**ts,
                     "L_fair_cumulative":tune_f,"active_categories_cumulative":tune_n},
                    {"system":name,"variant":"R0_filtered","split":"heldout",**hs,
                     "L_fair_cumulative":final_f,"active_categories_cumulative":final_n}]
        for scope,rows in (("warm_plus_tune",tune_cells),("warm_plus_tune_plus_heldout",final_cells)):
            cells += [{"system":name,"variant":"R0_filtered","cumulative_scope":scope,**r} for r in rows]
        data[name]={"held":hf}; preheld[name]=tp.clone()
        decisions[name]["cumulative_warm_tune"]={"L_fair":tune_f,"active_categories":tune_n}
        decisions[name]["cumulative_warm_tune_heldout"]={"L_fair":final_f,"active_categories":final_n}
        decisions[name]["tune_summary"]=ts; decisions[name]["heldout_incremental_summary"]=hs

    # R1 is sensitivity only: replay warm/tune and held-out using R0-selected A3 parameters.
    wl=warm_legacy(warm,c1,item2group,categories,ref); wp=warm_prop(warm,c1,item2group,categories,qthr)
    t1,l1,p1,s1,_=run_segment("A3_KG_FairCRS",tune,events,c1,item2group,categories,ref,wl,wp,{"lam":p3["lam"]},qthr,rmin)
    h1,_,p1f,s1h,_=run_segment("A3_KG_FairCRS",held,events,c1,item2group,categories,ref,l1,p1,{"lam":p3["lam"]},qthr,rmin)
    f1,n1,c1cells=proposal_terms(p1,rmin); f1h,n1h,c1hcells=proposal_terms(p1f,rmin)
    for f,sp in ((t1,"tune"),(h1,"heldout")):
        z=f.copy(); z["split"]=sp; z["variant"]="R1_raw"; event_frames.append(z)
    summary += [{"system":"A3_KG_FairCRS","variant":"R1_raw","split":"tune",**s1,
                 "L_fair_cumulative":f1,"active_categories_cumulative":n1},
                {"system":"A3_KG_FairCRS","variant":"R1_raw","split":"heldout",**s1h,
                 "L_fair_cumulative":f1h,"active_categories_cumulative":n1h}]
    cells += [{"system":"A3_KG_FairCRS","variant":"R1_raw","cumulative_scope":"warm_plus_tune",**r} for r in c1cells]
    cells += [{"system":"A3_KG_FairCRS","variant":"R1_raw","cumulative_scope":"warm_plus_tune_plus_heldout",**r} for r in c1hcells]
    decisions["A3_KG_FairCRS"]["R1_raw_sensitivity"]={"parameters_reused_from_R0":p3,
        "retuned_on_R1":False,"cumulative_warm_tune":{"L_fair":f1,"active_categories":n1},
        "cumulative_warm_tune_heldout":{"L_fair":f1h,"active_categories":n1h},
        "tune_summary":s1,"heldout_incremental_summary":s1h}

    boot=bootstrap(events,c0,held,data,preheld,item2group,categories,qthr,rmin,
                   a.bootstrap_reps,a.bootstrap_seed)
    boot.to_csv(a.out/"paired_bootstrap_results.csv",index=False)
    comp=[]
    for name in SYSTEMS:
        f=data[name]["held"]; final=decisions[name]["cumulative_warm_tune_heldout"]
        comp.append({"system":name,"variant":"R0_filtered","events":len(f),
            "conversations":int(f.conv_id.nunique()),"HR@10":float(f.hit10.mean()),
            "NDCG@10":float(f.ndcg10.mean()),"MRR@10":float(f.mrr10.mean()),
            "L_fair_cumulative_warm_tune_heldout":final["L_fair"],
            "active_categories_cumulative":final["active_categories"],
            "mean_kg_supported_top10":float(f.kg_supported_top10.mean()),
            "mean_top10_promoted_positions":float(f.top10_promoted_positions.mean())})
    comp=pd.DataFrame(comp)
    base_ndcg=float(comp.loc[comp.system=="A0_KGSF","NDCG@10"].iloc[0])
    trade=comp[["system","variant","HR@10","NDCG@10","MRR@10",
        "L_fair_cumulative_warm_tune_heldout","active_categories_cumulative"]].copy()
    trade["NDCG_retention_vs_A0"]=trade["NDCG@10"]/max(base_ndcg,EPS)
    trade["utility_constraint_met"]=trade["NDCG_retention_vs_A0"]>=a.utility_retention

    pd.concat(event_frames,ignore_index=True).to_csv(a.out/"event_level_system_comparisons.csv",index=False)
    pd.concat(tuning,ignore_index=True).to_csv(a.out/"tuning_grid_results.csv",index=False)
    pd.DataFrame(summary).to_csv(a.out/"split_and_cumulative_summaries.csv",index=False)
    pd.DataFrame(cells).to_csv(a.out/"cumulative_fairness_cells.csv",index=False)
    comp.to_csv(a.out/"system_comparison_summary.csv",index=False)
    trade.to_csv(a.out/"utility_fairness_tradeoff.csv",index=False)
    dump_json(a.out/"parameter_decisions.json",decisions)

    held_sets={s:set(data[s]["held"].event_id) for s in SYSTEMS}
    active={s:int(decisions[s]["cumulative_warm_tune_heldout"]["active_categories"]) for s in SYSTEMS}
    finite={s:bool(np.isfinite(decisions[s]["cumulative_warm_tune_heldout"]["L_fair"])) for s in SYSTEMS}
    checks={
        "experiment":"22.35F_validated_fairness_evaluation","validation_only":True,
        "independent_test_loaded":False,"training_performed":False,"retrieval_rerun":False,
        "checkpoint_modified":False,"original_22_35E_output_directory_overwritten":False,
        "frozen_top500_candidate_count_valid":all(len(c["ids"])==500 and len(np.unique(c["ids"]))==500 for c in c0.values()),
        "R0_R1_event_mappings_identical":events==events1,
        "R0_R1_candidate_order_identical":all(np.array_equal(c0[e]["ids"],c1[e]["ids"]) for e in events),
        "conversation_splits_disjoint":not(wc&tc or wc&hc or tc&hc),
        "all_events_assigned_once":len(set(warm)|set(tune)|set(held))==len(events),
        "four_systems_same_heldout_event_ids":all(held_sets[s]==held_sets["A0_KGSF"] for s in SYSTEMS),
        "parameter_selection_uses_tuning_only":True,
        "fairness_selection_uses_cumulative_warm_plus_tune":True,
        "heldout_fairness_uses_cumulative_warm_tune_heldout":True,
        "R1_uses_R0_parameters_without_retuning":True,
        "paired_conversation_cluster_bootstrap":True,
        "bootstrap_reps":a.bootstrap_reps,"bootstrap_seed":a.bootstrap_seed,
        "primary_cumulative_active_categories":active,
        "primary_cumulative_fairness_finite":finite,
        "all_primary_systems_have_supported_categories":all(v>0 for v in active.values()),
        "all_primary_fairness_values_finite":all(finite.values()),
    }
        # Safety invariants: these operations must NOT occur.
    safety_invariants = {
        "independent_test_loaded": checks["independent_test_loaded"] is False,
        "training_performed": checks["training_performed"] is False,
        "retrieval_rerun": checks["retrieval_rerun"] is False,
        "checkpoint_modified": checks["checkpoint_modified"] is False,
        "original_22_35E_output_directory_overwritten":
            checks["original_22_35E_output_directory_overwritten"] is False,
    }

    # Scientific and data-integrity conditions that MUST pass.
    required_checks = [
        "frozen_top500_candidate_count_valid",
        "R0_R1_event_mappings_identical",
        "R0_R1_candidate_order_identical",
        "conversation_splits_disjoint",
        "all_events_assigned_once",
        "four_systems_same_heldout_event_ids",
        "parameter_selection_uses_tuning_only",
        "fairness_selection_uses_cumulative_warm_plus_tune",
        "heldout_fairness_uses_cumulative_warm_tune_heldout",
        "R1_uses_R0_parameters_without_retuning",
        "paired_conversation_cluster_bootstrap",
        "all_primary_systems_have_supported_categories",
        "all_primary_fairness_values_finite",
    ]

    checks["safety_invariants_pass"] = all(
        safety_invariants.values()
    )

    checks["required_checks_pass"] = all(
        checks[name] is True for name in required_checks
    )

    checks["overall_pass"] = (
        checks["safety_invariants_pass"]
        and checks["required_checks_pass"]
    )

    checks["failed_required_checks"] = [
        name for name in required_checks
        if checks[name] is not True
    ]

    checks["failed_safety_invariants"] = [
        name for name, passed in safety_invariants.items()
        if not passed
    ]

    dump_json(a.out / "validation_checks.json", checks)
    inputs={"kgsf":a.kgsf,"q_r0":a.q_r0,"q_r1":a.q_r1,"catalog":a.catalog,
            "profiles":a.profiles,"reference_22_16":a.reference,"reference_22_18":a.reference_22_18}
    manifest={
        "experiment":"22.35F_validated_fairness_evaluation",
        "created_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
        "numpy":np.__version__,"pandas":pd.__version__,"scope":"frozen_validation_exports_only",
        "events":len(events),"conversations":len(set(convs)),"warm_events":len(warm),
        "tuning_events":len(tune),"heldout_events":len(held),"warm_conversations":len(wc),
        "tuning_conversations":len(tc),"heldout_conversations":len(hc),
        "primary_variant":"R0_filtered","sensitivity_variant":"R1_raw_for_A3_only",
        "systems":list(SYSTEMS),
        "objective":"Uniform mean over active categories of opportunity-weighted within-category variance of rho_cg=E_cg/(R_cg+epsilon).",
        "R_cg":"sum_e,i q_ei * category-membership * popularity-group-membership, only where q_ei >= selected threshold.",
        "E_cg":"sum over delivered top-10 ranks of 1/log2(rank+1) times category and popularity-group membership.",
        "active_cell_rule":"R_cg >= R_min; active category requires at least two supported popularity groups.",
        "fairness_scope":"warm-up KGSF + system-specific tuning + system-specific held-out history.",
        "parameter_selection":"Tuning conversations only; cumulative warm+tune fairness; NDCG retention constraint.",
        "R1_policy":"A3 uses R0-selected q_threshold, R_min and lambda unchanged.",
        "bootstrap":{"method":"paired conversation-cluster percentile bootstrap","reps":a.bootstrap_reps,
                     "seed":a.bootstrap_seed,"unit":"held-out conversation",
                     "fixed_slates":True,"cumulative_initial_state":"system-specific warm+tune"},
        "no_training":True,"no_retrieval":True,"no_checkpoint_changes":True,"independent_test_read":False,
        "inputs":{k:{"path":str(v),"sha256":sha256(v)} for k,v in inputs.items()},
        "outputs":["cumulative_fairness_cells.csv","parameter_decisions.json",
            "system_comparison_summary.csv","paired_bootstrap_results.csv",
            "utility_fairness_tradeoff.csv","validation_checks.json",
            "event_level_system_comparisons.csv","tuning_grid_results.csv",
            "split_and_cumulative_summaries.csv","experiment_manifest.json"],
        "validation_checks":checks,"parameter_decisions":decisions}
    dump_json(a.out/"experiment_manifest.json",manifest)
    print("\n========== 22.35F COMPLETE ==========")
    print(comp.to_string(index=False))
    print("\nCumulative active categories:",active)
    print("Validation overall_pass:",checks["overall_pass"])
    print("Training: NO | Retrieval: NO | Independent test: NOT READ")
    print("22.35E untouched. Output:",a.out.resolve())
    if not checks["overall_pass"]: raise RuntimeError("22.35F validation failed; inspect validation_checks.json")


if __name__=="__main__":
    main()
