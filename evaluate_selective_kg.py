"""Phase 5.3: validation-only, target-blind, relation-aware KG retrieval.

Keeps R0/R1 logic unchanged by importing the prior evaluation module.
R2 weights two-hop movie/context paths by relation agreement, per-relation
movie frequency, and neighbor degree. It also reports overlap-free recall.

All hyperparameters are exploratory and fixed in advance. Do not tune on test.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog
from evaluate_kg_retrieval import load_graphs, build_index, retrieve


def load_relation_graph(source_dir):
    """Raw DBpedia adjacency: entity -> neighbor -> set(original relation IDs).

    Match original KG's undirected interpretation; do not include self loops.
    """
    with (Path(source_dir) / 'dbpedia_subkg.json').open(encoding='utf-8') as fh:
        raw = json.load(fh)
    graph = defaultdict(lambda: defaultdict(set))
    for head, entries in raw.items():
        h = int(head)
        for rel, tail in entries:
            r, t = int(rel), int(tail)
            if h == t or r == 185:
                continue
            graph[h][t].add(r)
            graph[t][h].add(r)
    return graph


def build_selective_index(graph, movie_ids, hub_cutoff):
    """Inverted (neighbor,relation) -> movie set with relation-level IDF.

    Distinct from R1: shared-neighbor votes require matching relation IDs.
    This may lower recall; it is a hypothesis to test rather than guaranteed gain.
    """
    movies = set(map(int, movie_ids))
    evidence = defaultdict(set)
    relation_movies = defaultdict(set)
    for movie in movies:
        for neighbor, relations in graph.get(movie, {}).items():
            for rel in relations:
                evidence[(neighbor, rel)].add(movie)
                relation_movies[rel].add(movie)
    evidence = {key: mids for key, mids in evidence.items()
                if 0 < len(mids) <= hub_cutoff}
    n_movies = len(movies)
    relation_idf = {r: math.log1p(n_movies / (1 + len(mids)))
                    for r, mids in relation_movies.items()}
    return graph, movies, evidence, relation_idf


def retrieve_selective(context_entities, index, max_context=30):
    """Context-aware scores with relation agreement and degree damping.

    Important: never accepts or reads evaluation targets.
    """
    graph, movies, evidence, relation_idf = index
    context = list(dict.fromkeys(int(x) for x in context_entities))[-max_context:]
    scores = Counter()
    for entity in context:
        neighbors = graph.get(entity, {})
        if not neighbors:
            continue
        # Direct movie link, independent of relation agreement.
        for nbr, rels in neighbors.items():
            if nbr in movies:
                scores[nbr] += max(relation_idf.get(r, 0.0) for r in rels)
        # Shared attribute: the movie and context entity must use same relation.
        # The degree penalty discourages overly generic neighbor entities.
        for nbr, rels in neighbors.items():
            degree_penalty = 1.0 / math.sqrt(1.0 + len(graph.get(nbr, {})))
            for rel in rels:
                mids = evidence.get((nbr, rel))
                if not mids:
                    continue
                weight = relation_idf.get(rel, 0.0) * degree_penalty / math.log2(2 + len(mids))
                for movie in mids:
                    scores[movie] += weight
    # Remove items already mentioned in context only in a later controlled
    # ablation; retain here for fair comparison with existing R0/R1.
    return sorted(scores, key=lambda m: (-scores[m], m))


def counters_to_report(c, ks):
    n = c['events']
    out = {
        'events': n,
        'nonempty_context_events': c['nonempty_context'],
        'retrieval_nonempty_events': c['nonempty_retrieval'],
        'context_target_overlap_events': c['overlap'],
        'mean_candidates_per_event': c['candidate_sum'] / n if n else None,
    }
    for segment in ('overall', 'zero_shot', 'one_shot', 'other',
                    'nonoverlap', 'empty_context', 'nonempty_context'):
        denom = c[f'{segment}_events']
        row = {'events': denom}
        for k in ks:
            row[f'recall@{k}'] = c[f'{segment}_hit@{k}'] / denom if denom else None
        row['candidate_recall_all'] = c[f'{segment}_any'] / denom if denom else None
        out[segment] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    ap.add_argument('--catalog', default='catalog_audit_corrected.pkl')
    ap.add_argument('--raw-dir', default='data/dataset/redial/nltk')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--hub-cutoff', type=int, default=500)
    ap.add_argument('--out', default='results_corrected/kg_selective_validation.json')
    args = ap.parse_args()

    system, valid, _test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data['item_entity_ids'], system.device)
    movie_ids = list(map(int, catalog['movie_ids']))
    freq = {int(k): int(v) for k, v in catalog['meta']['freq_dict'].items()}
    graph_paths = load_graphs(side_data, Path(args.raw_dir))
    baseline_indexes = {
        key: build_index(adj, movie_ids, args.hub_cutoff)
        for key, adj in graph_paths.items()
    }
    selective = build_selective_index(load_relation_graph(args.raw_dir), movie_ids, args.hub_cutoff)
    approaches = {
        'R0_filtered': lambda ctx: retrieve(ctx, baseline_indexes['R0_filtered']),
        'R1_raw': lambda ctx: retrieve(ctx, baseline_indexes['R1_raw']),
        'R2_relation_selective': lambda ctx: retrieve_selective(ctx, selective),
    }
    ks = (50, 100, 500)
    counts = {name: Counter() for name in approaches}
    pad = int(valid.pad_entity_idx)
    n_events = 0
    with torch.no_grad():
        for batch in tqdm(valid.get_rec_data(batch_size=args.batch_size, shuffle=False),
                          desc='R0/R1/R2 validation'):
            for row, label in zip(batch[0].tolist(), batch[-1].tolist()):
                target = int(label)
                if target not in freq:
                    raise ValueError(f'Missing target frequency: {target}')
                ctx = [int(x) for x in row if int(x) != pad]
                category = 'zero_shot' if freq[target] == 0 else ('one_shot' if freq[target] == 1 else 'other')
                subgroups = ('overall', category,
                             'nonempty_context' if ctx else 'empty_context')
                if target not in ctx:
                    subgroups += ('nonoverlap',)
                n_events += 1
                for method_name, fn in approaches.items():
                    c = counts[method_name]
                    ranked = fn(ctx)
                    c['events'] += 1
                    c['nonempty_context'] += bool(ctx)
                    c['nonempty_retrieval'] += bool(ranked)
                    c['overlap'] += target in ctx
                    c['candidate_sum'] += len(ranked)
                    # One pass over ranked list avoids repeated O(N) searches.
                    rank_by_movie = {m: ix for ix, m in enumerate(ranked)}
                    rank = rank_by_movie.get(target, None)
                    for group in subgroups:
                        c[f'{group}_events'] += 1
                        if rank is not None:
                            c[f'{group}_any'] += 1
                            for k in ks:
                                if rank < k:
                                    c[f'{group}_hit@{k}'] += 1
    if n_events != 3010:
        raise RuntimeError(f'Expected 3010 validation events, got {n_events}')
    report = {
        'split': 'validation_only',
        'method_note': ('R2 uses matching raw relation IDs on shared two-hop paths, '
                        'inverse relation popularity and neighbor-degree damping; '
                        'all settings fixed, not optimized.'),
        'candidate_movies': len(movie_ids),
        'hub_cutoff': args.hub_cutoff,
        'important': ('R2 can perform worse than R1. Use validation to decide next methods; '
                      'reserve test for final, preregistered comparison.'),
        'results': {name: counters_to_report(c, ks) for name, c in counts.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()