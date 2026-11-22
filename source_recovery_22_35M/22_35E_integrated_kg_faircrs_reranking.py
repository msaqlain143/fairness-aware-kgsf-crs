#!/usr/bin/env python
"""Experiment 22.35E: validation-only integrated KG-FairCRS reranking.

Primary: R0_filtered q_ei. Sensitivity: R1_raw q_ei.
Four systems: frozen KGSF, 22.16 adaptive exposure, 22.18 contextual KG gate,
and proposal-driven KG-FairCRS.

No training, retrieval, checkpoint changes, or independent-test access.
"""
from __future__ import annotations
import argparse, ast, importlib.util, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

EPS, TOP_K = 1e-12, 10
UTILITY_RETENTION = 0.95
Q_GRID = (0.0005, 0.001, 0.002, 0.005)
LAM16_GRID = (0.05, 0.10, 0.20, 0.40, 0.80)
LAM18_GRID = (0.05, 0.10, 0.20, 0.40, 0.80)
TAU_GRID = (25.0, 50.0, 100.0, 250.0)
# R_min is a relevance-mass support threshold; selected only on tune.
RMIN_GRID = (0.0, 0.05, 0.10, 0.25, 0.50)
# Eq. (12) uses raw deficit units; lambda is therefore tuned on a log grid.
LAM_FAIR_GRID = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0)

D_KGSF = Path("results_corrected/experiment_22_35/kgsf_validation/kgsf_validation_top500.csv")
D_R0 = Path("results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R0_filtered.csv")
D_R1 = Path("results_corrected/experiment_22_35/relevance_validation/event_candidate_q_ei_R1_raw.csv")
D_CAT = Path("catalog_audit_corrected.pkl")
D_PROFILES = Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
D_22_16 = Path("22_16_category_context_utility_calibration.py")
D_22_18 = Path("22_18_contextual_relevance_gate_validation.py")
D_OUT = Path("results_corrected/experiment_22_35/integrated_reranking")


def parse_list(x, name):
    if isinstance(x, list): y = x
    elif isinstance(x, (tuple, np.ndarray)): y = list(x)
    else:
        try: y = json.loads(x)
        except Exception: y = ast.literal_eval(x)
    if not isinstance(y, list): raise ValueError(f"{name} must be a list")
    return y


def load_reference(path):
    spec = importlib.util.spec_from_file_location("ref_22_16", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    required = ("load_metadata", "split_conversations", "ExposureLedger",
                "category_context", "fairness_mad", "rerank")
    missing = [x for x in required if not hasattr(mod, x)]
    if missing: raise AttributeError(f"22.16 missing required API: {missing}")
    return mod


def load_22_18(path):
    """Import the existing 22.18 contextual gate without changing its code."""
    spec = importlib.util.spec_from_file_location("ref_22_18", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 22.18 reference script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "contextual_gate_rerank"):
        raise AttributeError("22.18 script lacks contextual_gate_rerank().")
    return mod


def load_data(kgsf_path, q_path, catalog):
    k = pd.read_csv(kgsf_path); qdf = pd.read_csv(q_path)
    reqk = {"event_id","conv_id","target_movie_id","top_movie_ids","top_scores"}
    reqq = {"event_id","candidate_rank","movie_id","kg_rank","q_ei"}
    if reqk-set(k): raise ValueError(f"KGSF export missing {sorted(reqk-set(k))}")
    if reqq-set(qdf): raise ValueError(f"{q_path} missing {sorted(reqq-set(qdf))}")
    if k.event_id.duplicated().any() or qdf.duplicated(["event_id","movie_id"]).any():
        raise ValueError("Duplicate event/candidate keys.")
    if set(k.event_id.astype(int)) != set(qdf.event_id.astype(int)):
        raise ValueError("KGSF and q_ei event IDs differ.")
    qmap = {int(e): f.sort_values("candidate_rank") for e,f in qdf.groupby("event_id")}
    events, cand = {}, {}
    for row in k.sort_values("event_id").itertuples(index=False):
        e = int(row.event_id)
        ids = np.asarray(parse_list(row.top_movie_ids,"top_movie_ids"),dtype=np.int64)
        scores = np.asarray(parse_list(row.top_scores,"top_scores"),dtype=np.float64)
        qrow = qmap[e]
        qids = qrow.movie_id.astype("int64").to_numpy()
        qrank = qrow.candidate_rank.astype("int64").to_numpy()
        q = pd.to_numeric(qrow.q_ei,errors="raise").to_numpy(float)
        kg = pd.to_numeric(qrow.kg_rank,errors="raise").to_numpy(int)
        if len(ids)!=500 or len(scores)!=500 or len(np.unique(ids))!=500:
            raise ValueError(f"Invalid frozen Top-500 at event {e}")
        if not np.array_equal(ids,qids) or not np.array_equal(qrank,np.arange(1,501)):
            raise ValueError(f"Candidate order mismatch in q_ei at event {e}")
        if not np.isfinite(scores).all() or not np.isfinite(q).all() or (q<0).any():
            raise ValueError(f"Non-finite or negative values at event {e}")
        if not np.isclose(q.sum(),1,atol=1e-8): raise ValueError(f"q_ei does not sum to 1 at {e}")
        if not set(map(int,ids)).issubset(catalog): raise ValueError(f"Out-of-catalog item at {e}")
        events[e]={"conv":int(row.conv_id),"target":int(row.target_movie_id)}
        cand[e]={"ids":ids,"scores":scores,"kg":kg,"q":q}
    if len(events)!=3010 or len({x["conv"] for x in events.values()})!=883:
        raise ValueError("Expected 3,010 validation events and 883 conversations.")
    return events,cand


def groups_for(c,item2group):
    return np.asarray([item2group[int(x)] for x in c["ids"]],dtype=int)


def metric(ids,target):
    pos=np.flatnonzero(np.asarray(ids[:TOP_K])==int(target))
    if not len(pos): return 0,0.0,0.0
    r=int(pos[0])+1
    return 1,1/math.log2(r+1),1/r


def update_legacy(ledger,c,ranked,item2group,categories):
    cg=groups_for(c,item2group)
    rg=np.asarray([item2group[int(x)] for x in ranked[:TOP_K]],dtype=int)
    ledger.update(c["ids"],cg,np.asarray(ranked[:TOP_K],dtype=np.int64),rg,categories)


def rank_22_16(c,item2group,categories,ledger,lam,ref):
    ids,_,info=ref.rerank(c,item2group,categories,ledger,float(lam),"category")
    return np.asarray(ids,dtype=np.int64),info


def rank_22_18(c,item2group,categories,ledger,lam,tau):
    # Call the original 22.18 function directly; no altered copy is used.
    ranked, info = ref18.contextual_gate_rerank(
        c, item2group, categories, ledger, float(lam), float(tau)
    )
    return np.asarray(ranked,dtype=np.int64), info


class ProposalLedger:
    """Proposal Eqs. (21)-(25): relevance mass R_cg and discounted exposure E_cg."""
    def __init__(self):
        self.R=defaultdict(float); self.E=defaultdict(float)

    def add_R(self,c,item2group,categories,q_threshold):
        gr=groups_for(c,item2group)
        for j,mid in enumerate(c["ids"]):
            if c["q"][j] < q_threshold: continue
            for cat in categories.get(int(mid),[]):
                self.R[(cat,int(gr[j]))]+=float(c["q"][j])

    def add_E(self,ranked,item2group,categories):
        for rank,mid in enumerate(ranked[:TOP_K],start=1):
            w=1/math.log2(rank+1)
            for cat in categories.get(int(mid),[]):
                self.E[(cat,int(item2group[int(mid)]))]+=w

    def clone(self):
        x=ProposalLedger(); x.R.update(self.R); x.E.update(self.E); return x


def proposal_terms(state,r_min):
    cats=defaultdict(list)
    for (cat,g),r in state.R.items():
        if r>EPS and r>=r_min: cats[cat].append((g,r))
    active={c:v for c,v in cats.items() if len(v)>=2}
    n=len(active); loss_by_cat={}; cells=[]
    if not n: return loss_by_cat,float("nan"),0,cells
    for cat,items in active.items():
        total=sum(r for _,r in items)
        p=np.asarray([r/total for _,r in items])
        rho=np.asarray([state.E[(cat,g)]/(r+EPS) for g,r in items])
        mean=float(np.sum(p*rho))
        var=float(np.sum(p*(rho-mean)**2))
        loss_by_cat[cat]=var/n
        for (g,r),pp,rr in zip(items,p,rho):
            cells.append({"category":cat,"popularity_group":int(g),"R_cg":r,
                          "E_cg":state.E[(cat,g)],"rho_cg":float(rr),"p_cg":float(pp)})
    return loss_by_cat,float(sum(loss_by_cat.values())),n,cells


def proposal_deficits(state,candidate,item2group,categories,r_min):
    """d_cg=max(0,rho_bar_c-rho_cg), using cumulative proposal R/E."""
    active=defaultdict(list)
    for (cat,g),r in state.R.items():
        if r>EPS and r>=r_min: active[cat].append((g,r))
    means={}
    for cat,items in active.items():
        if len(items)<2: continue
        total=sum(r for _,r in items)
        means[cat]=sum((r/total)*(state.E[(cat,g)]/(r+EPS)) for g,r in items)
    out={}
    for mid in candidate["ids"]:
        g=int(item2group[int(mid)])
        for cat in categories.get(int(mid),[]):
            if cat not in means: continue
            r=state.R[(cat,g)]
            if r<r_min or r<=EPS: continue
            rho=state.E[(cat,g)]/(r+EPS)
            out[(cat,g)]=max(0.0,means[cat]-rho)
    return out


def rank_integrated(c,item2group,categories,state,q_threshold,r_min,lam):
    # Add current event's relevance opportunity before calculating its deficits.
    state.add_R(c,item2group,categories,q_threshold)
    deficits=proposal_deficits(state,c,item2group,categories,r_min)
    ids,scores=c["ids"],c["scores"]
    adjusted=scores.copy()
    eligible=c["q"]>=q_threshold
    bonus=np.zeros(len(ids))
    for j,mid in enumerate(ids):
        if not eligible[j]: continue
        cats=categories.get(int(mid),[])
        if cats:
            # Equal membership averaging prevents multi-genre items receiving
            # a mechanically larger adjustment; A_ic remains binary per category.
            bonus[j]=float(np.mean([
                deficits.get((cat,int(item2group[int(mid)])),0.0) for cat in cats
            ]))
    adjusted[eligible]+=float(lam)*bonus[eligible]
    order=np.lexsort((ids,-adjusted))
    eligible_order=[int(j) for j in order if eligible[j]]
    ineligible_order=[int(j) for j in order if not eligible[j]]
    chosen=eligible_order[:TOP_K]
    fallback=max(0,TOP_K-len(chosen))
    # Proposal-consistent fallback: fill unfilled positions in frozen KGSF order;
    # below-threshold candidates are never fairness-promoted.
    if fallback:
        base_order=np.argsort(-scores,kind="stable")
        chosen_set=set(chosen)
        chosen.extend([int(j) for j in base_order if int(j) not in chosen_set][:fallback])
    ranked=ids[np.asarray(chosen[:TOP_K],dtype=int)]
    state.add_E(ranked,item2group,categories)
    return ranked,{"eligible_candidates":int(eligible.sum()),
        "eligible_relevance_mass":float(c["q"][eligible].sum()),
        "kg_supported_candidates":int(np.sum(c["kg"]>0)),
        "kg_supported_eligible_candidates":int(np.sum((c["kg"]>0)&eligible)),
        "fallback_positions":int(fallback),
        "top10_promoted_positions":int(np.sum(np.asarray(chosen[:TOP_K])!=np.arange(TOP_K)))}


def run_segment(name,event_ids,events,cands,item2group,categories,ref,
                start_legacy,start_prop,params,q_threshold=0.0,r_min=0.0,
                collect_frame=True):
    legacy=start_legacy.clone(); prop=start_prop.clone(); measure=ProposalLedger(); rows=[]
    for eid in sorted(event_ids):
        c=cands[eid]
        if name=="A0_KGSF":
            ranked=c["ids"]; info={}
        elif name=="A1_22_16":
            ranked,info=rank_22_16(c,item2group,categories,legacy,params["lam"],ref)
        elif name=="A2_22_18":
            ranked,info=rank_22_18(c,item2group,categories,legacy,params["lam"],params["tau"])
        elif name=="A3_KG_FairCRS":
            ranked,info=rank_integrated(c,item2group,categories,prop,q_threshold,r_min,params["lam"])
        else: raise ValueError(name)
        hit,ndcg,mrr=metric(ranked,events[eid]["target"])
        kgmap={int(m):int(k) for m,k in zip(c["ids"],c["kg"])}
        rows.append({"system":name,"event_id":eid,"conv_id":events[eid]["conv"],
            "hit10":hit,"ndcg10":ndcg,"mrr10":mrr,
            "kg_supported_candidates":int(np.sum(c["kg"]>0)),
            "kg_supported_top10":sum(kgmap.get(int(m),0)>0 for m in ranked[:TOP_K]),
            "eligible_candidates":int(info.get("eligible_candidates",0)),
            "fallback_positions":int(info.get("fallback_positions",0)),
            "top10_promoted_positions":int(info.get("top10_promoted_positions",0))})
        # A1/A2/A0 legacy ledger update occurs after delivery.
        update_legacy(legacy,c,ranked,item2group,categories)
        # Common proposal estimand: accumulate R from selected q threshold and
        # E from each system's delivered slate, separately from state used online.
        measure.add_R(c,item2group,categories,q_threshold)
        measure.add_E(ranked,item2group,categories)
        if name!="A3_KG_FairCRS":
            prop.add_R(c,item2group,categories,q_threshold); prop.add_E(ranked,item2group,categories)
    _,fair,ncat,cells=proposal_terms(measure,r_min)
    frame=pd.DataFrame(rows)
    summary={"events":len(frame),"conversations":int(frame.conv_id.nunique()),
        "HR@10":float(frame.hit10.mean()),"NDCG@10":float(frame.ndcg10.mean()),
        "MRR@10":float(frame.mrr10.mean()),"L_fair":fair,
        "active_categories":ncat,
        "mean_kg_supported_top10":float(frame.kg_supported_top10.mean()),
        "mean_eligible_candidates":float(frame.eligible_candidates.mean()),
        "mean_fallback_positions":float(frame.fallback_positions.mean()),
        "mean_top10_promoted_positions":float(frame.top10_promoted_positions.mean())}
    return frame,legacy,prop,summary,cells


def warm_legacy(ids,cands,item2group,categories,ref):
    ledger=ref.ExposureLedger()
    for eid in sorted(ids):
        c=cands[eid]; gr=groups_for(c,item2group)
        ledger.update(c["ids"],gr,c["ids"],gr,categories)
    return ledger


def warm_proposal(ids,cands,item2group,categories,threshold):
    state=ProposalLedger()
    for eid in sorted(ids):
        c=cands[eid]; state.add_R(c,item2group,categories,threshold)
        state.add_E(c["ids"],item2group,categories)
    return state


def tune_legacy(name,events,cands,warm_ids,tune_ids,item2group,categories,ref,retention,qthr,rmin):
    baseledger=warm_legacy(warm_ids,cands,item2group,categories,ref)
    _,_,_,base,_=run_segment("A0_KGSF",tune_ids,events,cands,item2group,categories,ref,
                             baseledger,ProposalLedger(),{},qthr,rmin)
    base_ndcg=max(base["NDCG@10"],EPS); trials=[]; best=None
    grid=([{"lam":x} for x in LAM16_GRID] if name=="A1_22_16" else
          [{"lam":x,"tau":t} for t in TAU_GRID for x in LAM18_GRID])
    for p in grid:
        frame,_,_,s,_=run_segment(name,tune_ids,events,cands,item2group,categories,ref,
                                  baseledger,ProposalLedger(),p,qthr,rmin)
        retention_value=s["NDCG@10"]/base_ndcg
        trials.append({**p,**s,"NDCG_retention":retention_value,
                       "utility_constraint_met":retention_value>=retention})
        if retention_value>=retention and np.isfinite(s["L_fair"]):
            key=(s["L_fair"],-s["NDCG@10"])
            if best is None or key<best[0]: best=(key,p)
    fallback={"lam":0.0} if name=="A1_22_16" else {"lam":0.0,"tau":TAU_GRID[0]}
    return (best[1] if best else fallback),pd.DataFrame(trials),base


def tune_integrated(events,cands,warm_ids,tune_ids,item2group,categories,ref,retention):
    baseledger=warm_legacy(warm_ids,cands,item2group,categories,ref)
    _,_,_,base,_=run_segment("A0_KGSF",tune_ids,events,cands,item2group,categories,ref,
                             baseledger,ProposalLedger(),{},0.0,0.0)
    base_ndcg=max(base["NDCG@10"],EPS); trials=[]; best=None
    for qthr in Q_GRID:
        for rmin in RMIN_GRID:
            for lam in LAM_FAIR_GRID:
                lp=warm_legacy(warm_ids,cands,item2group,categories,ref)
                pp=warm_proposal(warm_ids,cands,item2group,categories,qthr)
                _,_,_,s,_=run_segment("A3_KG_FairCRS",tune_ids,events,cands,item2group,
                    categories,ref,lp,pp,{"lam":lam},qthr,rmin)
                ret=s["NDCG@10"]/base_ndcg
                trials.append({"q_threshold":qthr,"R_min":rmin,"lambda":lam,**s,
                               "NDCG_retention":ret,"utility_constraint_met":ret>=retention})
                if ret>=retention and np.isfinite(s["L_fair"]):
                    key=(s["L_fair"],-s["NDCG@10"],qthr,rmin,lam)
                    if best is None or key<best[0]:
                        best=(key,{"q_threshold":qthr,"R_min":rmin,"lam":lam})
    if best: return best[1],pd.DataFrame(trials),"UTILITY_CONSTRAINED_MINIMUM_L_FAIR"
    return {"q_threshold":Q_GRID[0],"R_min":RMIN_GRID[0],"lam":0.0},pd.DataFrame(trials),\
           "NO_FEASIBLE_CONFIGURATION_FALLBACK_TO_KGSF_ORDER"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kgsf",type=Path,default=D_KGSF)
    ap.add_argument("--q-r0",type=Path,default=D_R0)
    ap.add_argument("--q-r1",type=Path,default=D_R1)
    ap.add_argument("--catalog",type=Path,default=D_CAT)
    ap.add_argument("--profiles",type=Path,default=D_PROFILES)
    ap.add_argument("--reference",type=Path,default=D_22_16)
    ap.add_argument("--reference-22-18",type=Path,default=D_22_18)
    ap.add_argument("--out",type=Path,default=D_OUT)
    ap.add_argument("--utility-retention",type=float,default=UTILITY_RETENTION)
    a=ap.parse_args()
    for p in (a.kgsf,a.q_r0,a.q_r1,a.catalog,a.profiles,a.reference,a.reference_22_18):
        if not p.is_file(): raise FileNotFoundError(p)
    ref=load_reference(a.reference)
    global ref18
    ref18=load_22_18(a.reference_22_18)
    item2group,catalog,categories=ref.load_metadata(a.catalog,a.profiles)
    events,c0=load_data(a.kgsf,a.q_r0,catalog)
    events1,c1=load_data(a.kgsf,a.q_r1,catalog)
    if events!=events1: raise RuntimeError("R0/R1 event mappings differ.")
    for eid in events:
        if not np.array_equal(c0[eid]["ids"],c1[eid]["ids"]):
            raise RuntimeError(f"R0/R1 candidate mismatch at {eid}")
    convs=[v["conv"] for v in events.values()]
    wc,tc,hc=ref.split_conversations(convs)
    if wc&tc or wc&hc or tc&hc: raise RuntimeError("Conversation split overlap.")
    warm=sorted(e for e,v in events.items() if v["conv"] in wc)
    tune=sorted(e for e,v in events.items() if v["conv"] in tc)
    held=sorted(e for e,v in events.items() if v["conv"] in hc)
    if set(warm)|set(tune)|set(held)!=set(events): raise RuntimeError("Split does not cover events.")
    a.out.mkdir(parents=True,exist_ok=True)
    print("="*76); print("EXPERIMENT 22.35E — INTEGRATED KG-FairCRS RERANKING"); print("="*76)
    print(f"Events={len(events)} | conversations={len(set(convs))} | warm/tune/held={len(wc)}/{len(tc)}/{len(hc)}")
    print("R0_filtered primary; R1_raw sensitivity; independent test is not read.")

    # Select the integrated threshold first. The same q threshold/R_min define
    # the proposal fairness estimand for all four systems.
    p3,grid3,status3=tune_integrated(events,c0,warm,tune,item2group,categories,ref,a.utility_retention)
    qthr,rmin=p3["q_threshold"],p3["R_min"]
    grid3.insert(0,"system","A3_KG_FairCRS"); grid3.to_csv(a.out/"tuning_A3_integrated.csv",index=False)

    decisions={"A3_KG_FairCRS":{"parameters":p3,"selection_status":status3}}
    all_tuning=[grid3]; all_events=[]; summaries=[]

    for name in ("A0_KGSF","A1_22_16","A2_22_18"):
        if name=="A0_KGSF": params={}
        else:
            params,grid,_=tune_legacy(name,events,c0,warm,tune,item2group,categories,ref,
                                      a.utility_retention,qthr,rmin)
            grid.insert(0,"system",name); all_tuning.append(grid)
        decisions[name]={"parameters":params}
        wl=warm_legacy(warm,c0,item2group,categories,ref)
        wp=warm_proposal(warm,c0,item2group,categories,qthr)
        tf,tl,tp,ts,tcells=run_segment(name,tune,events,c0,item2group,categories,ref,
                                       wl,wp,params,qthr,rmin)
        hf,hl,hp,hs,hcells=run_segment(name,held,events,c0,item2group,categories,ref,
                                       tl,tp,params,qthr,rmin)
        for f,sp in ((tf,"tune"),(hf,"heldout")):
            f=f.copy(); f["split"]=sp; f["variant"]="R0_filtered"; all_events.append(f)
        decisions[name]["tune_summary"]=ts; decisions[name]["heldout_summary"]=hs
        summaries.extend([{"system":name,"split":"tune",**ts},{"system":name,"split":"heldout",**hs}])

    # A3 selected configuration on R0, followed by R1 sensitivity with exactly
    # the same threshold, support threshold, and lambda (no R1 retuning).
    for variant,cands in (("R0_filtered",c0),("R1_raw",c1)):
        wl=warm_legacy(warm,cands,item2group,categories,ref)
        wp=warm_proposal(warm,cands,item2group,categories,qthr)
        tf,tl,tp,ts,tcells=run_segment("A3_KG_FairCRS",tune,events,cands,item2group,
            categories,ref,wl,wp,{"lam":p3["lam"]},qthr,rmin)
        hf,_,_,hs,hcells=run_segment("A3_KG_FairCRS",held,events,cands,item2group,
            categories,ref,tl,tp,{"lam":p3["lam"]},qthr,rmin)
        for f,sp in ((tf,"tune"),(hf,"heldout")):
            f=f.copy(); f["split"]=sp; f["variant"]=variant; all_events.append(f)
        decisions["A3_KG_FairCRS"][f"{variant}_tune_summary"]=ts
        decisions["A3_KG_FairCRS"][f"{variant}_heldout_summary"]=hs
        summaries.extend([{"system":"A3_KG_FairCRS","variant":variant,"split":"tune",**ts},
                          {"system":"A3_KG_FairCRS","variant":variant,"split":"heldout",**hs}])
        pd.DataFrame(hcells).to_csv(
            a.out/f"proposal_fairness_cells_{variant}_{'heldout'}.csv",index=False)

    pd.concat(all_events,ignore_index=True).to_csv(a.out/"event_level_system_comparisons.csv",index=False)
    pd.concat(all_tuning,ignore_index=True).to_csv(a.out/"tuning_grid_results.csv",index=False)
    pd.DataFrame(summaries).to_csv(a.out/"system_comparison_summary.csv",index=False)
    (a.out/"parameter_decisions.json").write_text(json.dumps(decisions,indent=2),encoding="utf-8")
    manifest={
        "experiment":"22.35E_integrated_kg_faircrs_reranking","split":"validation_only",
        "events":len(events),"conversations":len(set(convs)),
        "warm_conversations":len(wc),"tuning_conversations":len(tc),"heldout_conversations":len(hc),
        "primary_variant":"R0_filtered","sensitivity_variant":"R1_raw",
        "systems":["A0_KGSF","A1_22_16","A2_22_18","A3_KG_FairCRS"],
        "reused_implementations":{"22_16":str(a.reference),"22_18":str(a.reference_22_18)},
        "proposal_objective":"L_fair = sum_c omega_c sum_g p_cg (rho_cg-rho_bar_c)^2; uniform omega over active categories",
        "R_cg":"sum_e sum_i q_ei A_ic I(g(i)=g) I(i in I_e)",
        "E_cg":"sum_e sum_r 1/log2(r+1) A_pi_e[r],c I(g(pi_e[r])=g)",
        "score":"s_final = s_KGSF + lambda * category-averaged max(0,rho_bar_c-rho_cg), only for q_ei-eligible candidates",
        "eligibility":"q_ei >= threshold selected on tuning conversations; KG support measured but not a hard gate",
        "fallback":"if fewer than Top-10 eligible items, fill remaining slots in frozen KGSF order without fairness promotion",
        "R_min_and_q_threshold_selected_on":"tuning conversations only",
        "heldout_used_for_parameter_selection":False,"independent_test_read":False,
        "training_performed":False,"retrieval_performed":False,"checkpoint_modified":False,
        "outputs":["event_level_system_comparisons.csv","tuning_grid_results.csv",
                   "system_comparison_summary.csv","parameter_decisions.json",
                   "proposal_fairness_cells_R0_filtered_heldout.csv",
                   "proposal_fairness_cells_R1_raw_heldout.csv","experiment_manifest.json"],
        "decisions":decisions}
    (a.out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\n========== 22.35E COMPLETE ==========")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Selected integrated parameters:",p3,"|",status3)
    print("Training: NO | Retrieval: NO | Independent test: NOT READ")
    print("Output:",a.out.resolve())


if __name__=="__main__":
    main()
