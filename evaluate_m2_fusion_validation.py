"""Phase 5.4: validation-only KGSF + R1 KG rank-fusion diagnostic.

No test split, no retraining, no target use in retrieval, no change to M0/M1 scripts.
R1 returns candidate movie IDs ranked by graph score. Fuse reciprocals of
full movie-only KGSF rank and R1 rank; absent KG candidates get zero KG vote.
This is an exploratory rank-fusion baseline, NOT yet a novel KG-RAG method.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog
from evaluate_kg_retrieval import load_graphs, build_index, retrieve


def gini(x):
    a = np.sort(np.asarray(x, dtype=np.float64))
    if not len(a) or a.sum() == 0:
        return 0.0
    n = len(a)
    return float((2.0 * np.dot(np.arange(1, n + 1), a) / (n * a.sum())) - (n + 1.0) / n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    p.add_argument('--catalog', default='catalog_audit_corrected.pkl')
    p.add_argument('--raw-dir', default='data/dataset/redial/nltk')
    p.add_argument('--gpu', default='0')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--hub-cutoff', type=int, default=500)
    p.add_argument('--rrf-k', type=int, default=60)
    p.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    p.add_argument('--out', default='results_corrected/m2_rank_fusion_validation.json')
    args = p.parse_args()
    if args.rrf_k <= 0 or args.hub_cutoff <= 0 or any(a < 0 for a in args.alphas):
        raise ValueError('rrf-k, hub-cutoff must be positive; alphas nonnegative')

    system, valid, _test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data['item_entity_ids'], system.device)
    movie_ids = np.asarray(catalog['movie_ids'], dtype=np.int64)
    n_movies = len(movie_ids)
    pos = {int(m): i for i, m in enumerate(movie_ids)}
    freqs = {int(k): int(v) for k, v in catalog['meta']['freq_dict'].items()}
    raw_adj = load_graphs(side_data, Path(args.raw_dir))['R1_raw']
    index = build_index(raw_adj, movie_ids.tolist(), args.hub_cutoff)
    pad = int(valid.pad_entity_idx)
    group_ids = np.asarray(catalog['group_ids'], dtype=np.int64)
    alphas = list(dict.fromkeys(args.alphas))
    stats = {str(a): {'events': 0, 'hit10': 0, 'hit50': 0, 'ndcg10_sum': 0.0,
                      'zero_n': 0, 'zero_hit10': 0, 'one_n': 0, 'one_hit10': 0,
                      'exposures': np.zeros(n_movies, dtype=np.int64),
                      'group_counts': np.zeros(5, dtype=np.int64)} for a in alphas}
    nonempty_context = 0
    n_events = 0
    system.model.eval()
    with torch.no_grad():
        for batch in tqdm(valid.get_rec_data(batch_size=args.batch_size, shuffle=False),
                          desc='M2 R1 rank-fusion validation'):
            model_batch = tuple(t.to(system.device) for t in batch)
            output = system.model.recommend(model_batch, mode='test')
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError('KGSF.recommend() output is not the expected three-tuple')
            scores = output[2]
            scores = scores.index_select(1, catalog['candidate_ids']).detach().cpu().numpy()
            if scores.ndim != 2 or scores.shape[1] != n_movies:
                raise RuntimeError('Movie score shape mismatch')
            for context_row, target, row_scores in zip(batch[0].tolist(), batch[-1].tolist(), scores):
                target = int(target)
                if target not in pos:
                    raise RuntimeError(f'Ground truth movie not in eligible catalog: {target}')
                context = [int(x) for x in context_row if int(x) != pad]
                nonempty_context += int(bool(context))
                kg_ranked = retrieve(context, index)
                # Lexicographic tie breaker makes rank deterministic across NumPy versions.
                kgsf_order = np.lexsort((movie_ids, -row_scores))
                kgsf_rank = np.empty(n_movies, dtype=np.int32)
                kgsf_rank[kgsf_order] = np.arange(1, n_movies + 1, dtype=np.int32)
                kgsf_vote = 1.0 / (args.rrf_k + kgsf_rank.astype(np.float64))
                kg_vote = np.zeros(n_movies, dtype=np.float64)
                for rank, movie in enumerate(kg_ranked, 1):
                    local = pos.get(int(movie))
                    if local is not None:
                        kg_vote[local] = 1.0 / (args.rrf_k + rank)
                true_local = pos[target]
                for alpha in alphas:
                    s = stats[str(alpha)]
                    fused = kgsf_vote + alpha * kg_vote
                    order = np.lexsort((movie_ids, -fused))[:50]
                    where = np.flatnonzero(order == true_local)
                    rank = int(where[0]) + 1 if len(where) else None
                    s['events'] += 1
                    if rank is not None:
                        s['hit50'] += 1
                        if rank <= 10:
                            s['hit10'] += 1
                            s['ndcg10_sum'] += 1.0 / np.log2(rank + 1)
                    frequency = freqs[target]
                    if frequency == 0:
                        s['zero_n'] += 1
                        s['zero_hit10'] += int(rank is not None and rank <= 10)
                    if frequency == 1:
                        s['one_n'] += 1
                        s['one_hit10'] += int(rank is not None and rank <= 10)
                    top10 = order[:10]
                    np.add.at(s['exposures'], top10, 1)
                    np.add.at(s['group_counts'], group_ids[top10], 1)
                n_events += 1
    if n_events != 3010:
        raise RuntimeError(f'Expected 3010 validation recommendation events, got {n_events}')
    results = []
    for alpha in alphas:
        s = stats[str(alpha)]
        slots = int(s['exposures'].sum())
        if slots != n_events * 10 or s['group_counts'].sum() != slots:
            raise RuntimeError('Exposure accounting mismatch')
        results.append({
            'alpha': alpha, 'events': n_events, 'hit10': s['hit10']/n_events,
            'hit50': s['hit50']/n_events, 'ndcg10': s['ndcg10_sum']/n_events,
            'zero_shot_events': s['zero_n'],
            'zero_shot_hit10': s['zero_hit10']/s['zero_n'] if s['zero_n'] else None,
            'one_shot_events': s['one_n'],
            'one_shot_hit10': s['one_hit10']/s['one_n'] if s['one_n'] else None,
            'item_gini': gini(s['exposures']),
            'coverage10': float(np.count_nonzero(s['exposures'])/n_movies),
            'group_exposure_shares': (s['group_counts']/slots).tolist()
        })
    report = {
        'split': 'validation_only', 'method': 'R1 raw KG + KGSF movie-rank reciprocal-rank fusion',
        'formula': '1/(rrf_k + rank_KGSF) + alpha/(rrf_k + rank_KG), absent KG vote=0',
        'uses_target_for_retrieval': False, 'changes_checkpoint': False,
        'candidate_catalog': n_movies, 'events': n_events,
        'nonempty_context_events': nonempty_context,
        'hub_cutoff': args.hub_cutoff, 'rrf_k': args.rrf_k,
        'results': results,
        'caution': ('Use validation only to select alpha and utility constraints; '
                    'no test evaluation or novelty claim follows from this diagnostic.')
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', path.resolve())


if __name__ == '__main__':
    main()
