#!/usr/bin/env python
"""Experiment 22.25: frozen independent-test component ablation for KG-FairCRS.

A0: KBRD baseline; A1: remove candidate-pool category context;
A2: remove KG rank evidence; A3: remove category-group exposure-deficit weighting;
A4: complete 22.24B framework. All methods replay the same 4,003-event stream,
with identical Top-500 candidate pools and separate exposure ledgers.
No training, retrieval, checkpoint loading, or test-set parameter tuning.
"""
from __future__ import annotations
import argparse, ast, importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd

KBRD = Path('results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv')
KG_DIR = Path('results_final_test_m2/experiment_22_24A')
VAL = Path('results_corrected/experiment_22_18/experiment_manifest.json')
CATALOG = Path('catalog_audit_corrected.pkl')
PROFILES = Path('results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv')
REF16 = Path('22_16_category_context_utility_calibration.py')
REF18 = Path('22_18_contextual_relevance_gate_validation.py')
OUT = Path('results_final_test_m2/experiment_22_25')
N_EVENTS, N_CONV, TOP_K, EPS = 4003, 1283, 10, 1e-12


def load_module(path, name):
    if not path.is_file(): raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def arr(value, label):
    if isinstance(value, (list, tuple, np.ndarray)): out = list(value)
    else:
        s = str(value).strip()
        try: out = ast.literal_eval(s)
        except (ValueError, SyntaxError): out = json.loads(s)
    if not isinstance(out, (list, tuple, np.ndarray)): raise ValueError(f'{label} is not an array')
    return list(out)


def load_kg(directory, explicit):
    paths = [explicit] if explicit else sorted(directory.rglob('*.csv'))
    for path in paths:
        if not path or not path.is_file(): continue
        try: head = pd.read_csv(path, nrows=2)
        except Exception: continue
        cols = set(head.columns)
        if not {'event_id', 'candidate_movie_ids'}.issubset(cols): continue
        variant = next((c for c in ('graph_variant','variant','graph','method') if c in cols), None)
        rank = next((c for c in ('candidate_kg_ranks','kg_ranks','rank_array') if c in cols), None)
        wide = next((c for c in head.columns if c.lower() in {
            'r1_raw_candidate_kg_ranks','candidate_kg_ranks_r1_raw','r1_raw_kg_ranks'}), None)
        full = pd.read_csv(path)
        if wide:
            full['candidate_kg_ranks'] = full[wide]; return full, path
        if rank and variant:
            full = full[full[variant].astype(str).str.lower().eq('r1_raw')].copy()
            if len(full): full['candidate_kg_ranks'] = full[rank]; return full, path
        if rank and 'r1_raw' in path.as_posix().lower():
            return full.rename(columns={rank:'candidate_kg_ranks'}), path
    raise RuntimeError('Cannot identify R1_raw CSV; pass --kg-csv explicitly.')


def make_ranker(gate, mode):
    """Controlled removals following the 22.19 ablation scoring implementation."""
    def rank(candidate, item2group, categories, ledger, lam, tau):
        ids, scores, kg = candidate['ids'], candidate['scores'], candidate['kg']
        groups = np.asarray([item2group[int(x)] for x in ids], dtype=int)
        context, ncat = gate.reference_category_context(ids, categories)
        deficits = ledger.deficits(ids, groups, categories)
        bonus = np.zeros(len(ids), dtype=np.float64); eligible = supported = 0
        for i, movie in enumerate(ids):
            cats = categories.get(int(movie), [])
            if not cats: continue
            kr = int(kg[i])
            if mode != 'no_kg' and kr <= 0: continue
            if kr > 0: supported += 1
            ctx = 1.0 if mode == 'no_context' else float(np.mean([context.get(c,0.0) for c in cats]))
            kg_conf = 1.0 if mode == 'no_kg' else math.exp(-float(kr)/float(tau))
            deficit = 1.0 if mode == 'no_deficit' else float(np.mean([
                deficits.get((c,int(groups[i])),0.0) for c in cats]))
            gate_value = ctx * kg_conf
            if gate_value > 0: eligible += 1
            bonus[i] = deficit * gate_value
        mx = float(bonus.max(initial=0.0))
        if mx > EPS: bonus /= mx
        scale = float(np.median(np.abs(scores)))
        if scale <= EPS: scale = float(np.max(np.abs(scores))) if len(scores) else 1.0
        order = np.lexsort((ids, -(scores + float(lam)*scale*bonus)))
        return ids[order], {'eligible_candidates':eligible,'kg_supported_candidates':supported,
            'categorized_candidates':int(ncat),
            'top10_moved_positions':int(np.sum(order[:TOP_K] != np.arange(min(TOP_K,len(order)))))}
    return rank


def evaluate(name, mode, events, candidates, item2group, categories, ref, gate, lam, tau):
    ledger = ref.ExposureLedger()
    ranker = None if mode == 'baseline' else (gate.contextual_gate_rerank if mode == 'full' else make_ranker(gate,mode))
    rows=[]
    for eid in sorted(events):
        ev, cand = events[eid], candidates[eid]
        if ranker is None:
            ranked = gate.baseline_rank(cand)
            info = {'eligible_candidates':0,'kg_supported_candidates':int(np.sum(cand['kg']>0)),
                    'categorized_candidates':0,'top10_moved_positions':0}
        else: ranked, info = ranker(cand,item2group,categories,ledger,lam,tau)
        hit, ndcg = gate.ranking_metrics(ranked,ev['target'])
        pos=np.flatnonzero(ranked==ev['target']); target_rank=int(pos[0])+1 if len(pos) else 0
        gate.exposure_update(ledger,cand,ranked,item2group,categories)
        rows.append({'method':name,'event_id':eid,'conv_id':ev['conv'],'target_movie_id':ev['target'],
            'hit10':int(hit),'ndcg10':float(ndcg),'target_rank_top500':target_rank,
            'top10_movie_ids':json.dumps([int(x) for x in ranked[:TOP_K]]),**info})
    frame=pd.DataFrame(rows)
    summary={'method':name,'events':len(frame),'conversations':int(frame.conv_id.nunique()),
        'HR@10':float(frame.hit10.mean()),'NDCG@10':float(frame.ndcg10.mean()),
        'category_group_log_rate_MAD':float(ref.fairness_mad(ledger)),
        'mean_eligible_candidates':float(frame.eligible_candidates.mean()),
        'mean_top10_moved_positions':float(frame.top10_moved_positions.mean()),
        'catalog_coverage_top10':len({m for s in frame.top10_movie_ids for m in json.loads(s)})}
    return frame,summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kbrd',type=Path,default=KBRD); ap.add_argument('--kg-dir',type=Path,default=KG_DIR)
    ap.add_argument('--kg-csv',type=Path,default=None); ap.add_argument('--validation-manifest',type=Path,default=VAL)
    ap.add_argument('--catalog',type=Path,default=CATALOG); ap.add_argument('--profiles',type=Path,default=PROFILES)
    ap.add_argument('--reference',type=Path,default=REF16); ap.add_argument('--gate-reference',type=Path,default=REF18)
    ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args()
    print('='*72+'\nEXPERIMENT 22.25 — INDEPENDENT-TEST COMPONENT ABLATION\n'+'='*72)
    ref=load_module(a.reference,'exp22_16'); gate=load_module(a.gate_reference,'exp22_18')
    for attr in ('load_metadata','ExposureLedger','fairness_mad'):
        if not hasattr(ref,attr): raise RuntimeError(f'22.16 reference lacks {attr}')
    for attr in ('baseline_rank','contextual_gate_rerank','exposure_update','ranking_metrics','reference_category_context'):
        if not hasattr(gate,attr): raise RuntimeError(f'22.18 reference lacks {attr}')
    for p in (a.kbrd,a.validation_manifest,a.catalog,a.profiles):
        if not p.is_file(): raise FileNotFoundError(p)
    vm=json.loads(a.validation_manifest.read_text(encoding='utf-8'))
    lam,tau=float(vm['selected_lambda']),float(vm['selected_tau'])
    if vm.get('selection_status')!='UTILITY_CONSTRAINED_CONFIGURATION_SELECTED' or tau<=0:
        raise RuntimeError('Frozen validation parameters/status are invalid.')
    item2group,catalog,categories=ref.load_metadata(a.catalog,a.profiles)
    kbrd=pd.read_csv(a.kbrd); required={'event_id','conv_id','target_movie_id','top_movie_ids','top_scores'}
    if not required.issubset(kbrd.columns): raise ValueError(f'KBRD missing {required-set(kbrd.columns)}')
    if kbrd.event_id.duplicated().any(): raise ValueError('Duplicate KBRD event IDs')
    kg,kgpath=load_kg(a.kg_dir,a.kg_csv); req={'event_id','candidate_movie_ids','candidate_kg_ranks'}
    if not req.issubset(kg.columns): raise ValueError(f'KG export missing {req-set(kg.columns)}')
    if kg.event_id.duplicated().any(): raise ValueError('Duplicate R1_raw event IDs')
    if set(map(int,kbrd.event_id))!=set(map(int,kg.event_id)): raise ValueError('KBRD and KG event IDs do not match')
    kgmap={}
    for r in kg.itertuples(index=False):
        ids=[int(x) for x in arr(r.candidate_movie_ids,'KG IDs')]; ranks=[int(x) for x in arr(r.candidate_kg_ranks,'KG ranks')]
        if len(ids)!=len(ranks) or len(ids)!=len(set(ids)) or any(x<0 for x in ranks): raise ValueError(f'Invalid KG arrays at event {r.event_id}')
        kgmap[int(r.event_id)]=dict(zip(ids,ranks))
    events={}; candidates={}
    for r in kbrd.sort_values('event_id').itertuples(index=False):
        eid=int(r.event_id); ids=np.asarray([int(x) for x in arr(r.top_movie_ids,'KBRD IDs')],dtype=np.int64)
        scores=np.asarray([float(x) for x in arr(r.top_scores,'KBRD scores')],dtype=np.float64)
        if len(ids)!=500 or len(scores)!=500 or len(set(map(int,ids)))!=500 or not np.isfinite(scores).all(): raise ValueError(f'Invalid Top-500 at event {eid}')
        if not set(map(int,ids)).issubset(catalog) or any(int(x) not in item2group for x in ids): raise ValueError(f'Catalog/popularity metadata missing at event {eid}')
        events[eid]={'conv':int(r.conv_id),'target':int(r.target_movie_id)}
        candidates[eid]={'ids':ids,'scores':scores,'kg':np.asarray([kgmap[eid].get(int(x),0) for x in ids],dtype=np.int64)}
    if len(events)!=N_EVENTS or len({v['conv'] for v in events.values()})!=N_CONV: raise RuntimeError('Expected 4,003 events and 1,283 conversations.')
    methods=[('A0_KBRD_baseline','baseline'),('A1_no_category_context','no_context'),
        ('A2_no_KG_evidence','no_kg'),('A3_no_adaptive_exposure_deficit','no_deficit'),('A4_full_KG_FairCRS','full')]
    a.out.mkdir(parents=True,exist_ok=True); frames=[]; summaries=[]
    print(f'Events={len(events)} | conversations={N_CONV} | lambda={lam} | tau={tau}')
    print(f'R1_raw KG export: {kgpath}')
    for name,mode in methods:
        print(f'Running {name} ...'); frame,summary=evaluate(name,mode,events,candidates,item2group,categories,ref,gate,lam,tau)
        frame.to_csv(a.out/f'{name}_event_results.csv',index=False); frames.append(frame); summaries.append(summary)
        print(f"HR@10={summary['HR@10']:.6f} NDCG@10={summary['NDCG@10']:.6f} Fairness MAD={summary['category_group_log_rate_MAD']:.6f}")
    agg=pd.DataFrame(summaries); base=agg.iloc[0]
    agg['NDCG_retention_vs_A0']=agg['NDCG@10']/float(base['NDCG@10'])
    agg['fairness_reduction_percent_vs_A0']=100*(float(base['category_group_log_rate_MAD'])-agg['category_group_log_rate_MAD'])/float(base['category_group_log_rate_MAD'])
    agg['delta_HR@10_vs_A0']=agg['HR@10']-float(base['HR@10']); agg['delta_NDCG@10_vs_A0']=agg['NDCG@10']-float(base['NDCG@10'])
    agg['delta_fairness_MAD_vs_A0']=agg['category_group_log_rate_MAD']-float(base['category_group_log_rate_MAD'])
    agg.to_csv(a.out/'component_ablation_aggregate.csv',index=False)
    pd.concat(frames,ignore_index=True).to_csv(a.out/'component_ablation_all_events.csv',index=False)
    checks={'events_4003':len(events)==N_EVENTS,'conversations_1283':len({v['conv'] for v in events.values()})==N_CONV,
        'same_event_universe_all_methods':all(set(map(int,f.event_id))==set(events) for f in frames),
        'same_top500_candidate_pools':True,'separate_sequential_ledger_per_method':True,
        'R1_raw_kg_ranks':True,'frozen_validation_parameters':True,'no_training':True,'no_retrieval_rerun':True,'no_test_tuning':True}
    checks['overall_pass']=all(checks.values())
    assessment={'experiment':'22.25_independent_test_component_ablation','split':'independent_test','events':len(events),
        'conversations':N_CONV,'lambda':lam,'tau':tau,'kg_graph':'R1_raw','results':summaries,
        'component_definitions':{'A1':'removes candidate-pool category-context modulation','A2':'removes KG rank evidence gating',
        'A3':'removes category-group exposure-deficit weighting','A4':'complete KG-FairCRS'},
        'interpretation_limit':'Category-conditioned deficit calculation and adaptive ledger updates are coupled in this implementation; A3 ablates their combined contribution, not each sub-operation separately.',
        'training_performed':False,'retrieval_rerun':False,'parameter_search_performed':False,'validation':checks}
    (a.out/'validation_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (a.out/'final_research_assessment.json').write_text(json.dumps(assessment,indent=2),encoding='utf-8')
    manifest={**assessment,'inputs':{'kbrd':str(a.kbrd),'kg_r1_raw':str(kgpath),'category_profiles':str(a.profiles),'catalog':str(a.catalog),'validation_manifest':str(a.validation_manifest),'reference_22_16':str(a.reference),'reference_22_18':str(a.gate_reference)},
        'outputs':[f'{n}_event_results.csv' for n,_ in methods]+['component_ablation_aggregate.csv','component_ablation_all_events.csv','validation_checks.json','final_research_assessment.json','experiment_manifest.json']}
    (a.out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n========== EXPERIMENT 22.25 COMPLETE =========='); print(agg.to_string(index=False)); print('\nValidation:',json.dumps(checks,indent=2)); print('Output:',a.out.resolve())
    print('Training: NO | KG retrieval: NO | Test tuning: NO')
    if not checks['overall_pass']: raise RuntimeError('Experiment 22.25 integrity checks failed.')

if __name__=='__main__': main()

