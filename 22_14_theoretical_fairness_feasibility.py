#!/usr/bin/env python
"""Experiment 22.14: synthetic exhaustive verification of P3/P4.

No ReDial data, validation/test exports, model training, or KG retrieval is used.
Synthetic q_i values are fixtures, not empirical relevance estimates.
"""
import argparse, itertools, json, math
from pathlib import Path
import pandas as pd

EPS = 1e-12

def w(rank):
    return 1.0 / math.log2(rank + 1)

def catalog():
    raw = [
        ("a", .95, 0, {"drama": 1.}, .98, 1),
        ("b", .90, 0, {"drama": 1.}, .91, 0),
        ("c", .92, 1, {"drama": 1.}, .86, 0),
        ("d", .85, 1, {"drama": 1.}, .78, 0),
        ("e", .88, 0, {"comedy": 1.}, .83, 0),
        ("f", .91, 1, {"comedy": 1.}, .80, 0),
        ("g", .40, 0, {"drama": .5, "comedy": .5}, .70, 0),
        ("h", .30, 1, {"comedy": 1.}, .65, 0),
    ]
    out = []
    for iid, q, group, cats, score, target in raw:
        z = sum(cats.values())
        out.append(dict(item_id=iid, q=q, group=group,
                        cats={c:v/z for c,v in cats.items()},
                        score=score, target=target))
    return out

def metrics(slate, eligible, rmin):
    cats = sorted({c for x in eligible for c in x["cats"]})
    groups = sorted({x["group"] for x in eligible})
    R = {(c,g):0. for c in cats for g in groups}
    E = {(c,g):0. for c in cats for g in groups}
    for x in eligible:
        for c,a in x["cats"].items():
            R[c,x["group"]] += x["q"]*a
    for rank,x in enumerate(slate,1):
        for c,a in x["cats"].items():
            E[c,x["group"]] += w(rank)*a
    loss, detail = 0., {}
    for c in cats:
        active = [g for g in groups if R[c,g] >= rmin]
        if len(active) < 2:
            detail[c] = {"active_groups":active,"excluded":True}
            continue
        total = sum(R[c,g] for g in active)
        p = {g:R[c,g]/total for g in active}
        rho = {g:E[c,g]/(R[c,g]+EPS) for g in active}
        mean = sum(p[g]*rho[g] for g in active)
        lc = sum(p[g]*(rho[g]-mean)**2 for g in active)
        loss += lc
        detail[c] = {"active_groups":active,
                     "relevance_weights":{str(g):p[g] for g in active},
                     "normalized_exposure":{str(g):rho[g] for g in active},
                     "category_loss":lc,"excluded":False}
    return loss, detail

def utility(slate):
    for rank,x in enumerate(slate,1):
        if x["target"]:
            return 1/math.log2(rank+1)
    return 0.

def enumerate_case(items,k,qmin,rmin,umin,delta):
    eligible = [x for x in items if x["q"] >= qmin]
    if len(eligible) < k:
        raise ValueError("Eligible catalog smaller than K.")
    rows = []
    for slate in itertools.permutations(eligible,k):
        L, _ = metrics(slate,eligible,rmin)
        U = utility(slate)
        rows.append({"slate":"|".join(x["item_id"] for x in slate),
                     "fairness_loss":L,"ndcg":U,
                     "utility_feasible":U+EPS>=umin,
                     "fairness_feasible":L<=delta+EPS,
                     "jointly_feasible":U+EPS>=umin and L<=delta+EPS})
    frame = pd.DataFrame(rows)
    uf = frame[frame.utility_feasible]
    Lmin = None if uf.empty else float(uf.fairness_loss.min())
    return {
        "eligible_items":len(eligible),"enumerated_slates":len(frame),
        "U_min":umin,"delta":delta,
        "L_min_subject_to_utility":Lmin,
        "feasible":bool(Lmin is not None and Lmin<=delta+EPS),
        "jointly_feasible_slates":int(frame.jointly_feasible.sum()),
        "minimum_loss_any_slate":float(frame.fairness_loss.min()),
        "maximum_ndcg":float(frame.ndcg.max())
    }, frame

def margin_check(eligible):
    ordered=sorted(eligible,key=lambda x:x["score"],reverse=True)
    bonus={x["item_id"]:(.015 if x["group"]==1 else 0.) for x in ordered}
    gaps=[ordered[i]["score"]-ordered[i+1]["score"] for i in range(len(ordered)-1)]
    bvals=[bonus[x["item_id"]] for x in ordered]
    min_gap=min(gaps) if gaps else float("inf")
    max_diff=max(bvals)-min(bvals) if bvals else 0.
    return {"minimum_adjacent_base_score_gap":min_gap,
            "maximum_pairwise_bonus_difference":max_diff,
            "sufficient_order_preservation_condition":bool(min_gap>max_diff),
            "meaning":"If every adjacent base-score gap exceeds the maximum pairwise bonus difference, the complete ranking order is preserved."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("results_corrected/experiment_22_14"))
    ap.add_argument("--q-threshold",type=float,default=.5)
    ap.add_argument("--r-min",type=float,default=.1)
    ap.add_argument("--u-min",type=float,default=.5)
    ap.add_argument("--delta",type=float,default=.05)
    a=ap.parse_args()
    if not 0<=a.q_threshold<=1 or a.r_min<0 or not 0<=a.u_min<=1 or a.delta<0:
        raise ValueError("Invalid threshold or constraint.")
    items=catalog()
    specs=[("primary",a.u_min),("strict_utility",max(a.u_min,.95))]
    summaries=[]; frames=[]
    for name,umin in specs:
        s,f=enumerate_case(items,3,a.q_threshold,a.r_min,umin,a.delta)
        s["case"]=name; summaries.append(s)
        f.insert(0,"case",name); frames.append(f)
    a.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(a.out/"feasibility_case_summary.csv",index=False)
    pd.concat(frames,ignore_index=True).to_csv(a.out/"exhaustive_slate_results.csv",index=False)
    p4=margin_check([x for x in items if x["q"]>=a.q_threshold])
    (a.out/"p4_score_margin_check.json").write_text(json.dumps(p4,indent=2),encoding="utf-8")
    checks={
        "synthetic_data_only":True,
        "no_dataset_or_validation_test_exports_loaded":True,
        "no_training_or_retrieval":True,
        "all_slates_are_unique_permutations":all(len(s.split("|"))==len(set(s.split("|"))) for f in frames for s in f.slate),
        "finite_feasibility_characterization_reconciled":all(
            (s["L_min_subject_to_utility"] is None and not s["feasible"]) or
            (s["L_min_subject_to_utility"] is not None and s["feasible"]==(s["L_min_subject_to_utility"]<=a.delta+EPS))
            for s in summaries),
        "p4_sufficient_condition_computed":True
    }
    checks["overall_pass"]=all(checks.values())
    manifest={
        "experiment":"22.14",
        "name":"Finite-Slate Fairness Feasibility and Utility Preservation",
        "proposal_targets":["P3","P4"],
        "scope":"synthetic_exhaustive_verification",
        "real_dataset_used":False,"training_performed":False,
        "retrieval_rerun":False,"test_evaluation":False,
        "assumptions":["finite candidate catalog","fixed synthetic relevance q_i",
          "normalized multi-category memberships","fixed popularity groups",
          "single relevant target","logarithmic position exposure",
          "active cells satisfy R_cg >= R_min"],
        "cautions":["Synthetic q_i are not empirical relevance estimates.",
          "Enumeration verifies finite examples, not a universal theorem.",
          "The score-margin condition is sufficient, not necessary.",
          "The finite-set minimum characterization alone is not claimed as novel."],
        "outputs":["feasibility_case_summary.csv","exhaustive_slate_results.csv",
          "p4_score_margin_check.json","validation_checks.json","experiment_manifest.json"]}
    (a.out/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    (a.out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("EXPERIMENT 22.14")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("P4:",json.dumps(p4))
    print("VALIDATION:",json.dumps(checks,indent=2))
    print("OUTPUT:",a.out.resolve())
    if not checks["overall_pass"]:
        raise SystemExit("Validation failed.")

if __name__=="__main__":
    main()
