"""Phase 5.2: validation-only, target-blind graph candidate retrieval diagnostic.

Compares processed KGSF DBpedia graph (R0) with the raw DBpedia subgraph (R1).
Does not train, alter checkpoints, or use the test split.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog


def load_graphs(side_data, source_dir):
    raw_path = source_dir / 'dbpedia_subkg.json'
    if not raw_path.is_file():
        raise FileNotFoundError(f'Could not find {raw_path}')
    with raw_path.open(encoding='utf-8') as fh:
        raw = json.load(fh)
    raw_adj = defaultdict(set)
    for head, entries in raw.items():
        h = int(head)
        for relation, tail in entries:
            t = int(tail)
            if h != t and int(relation) != 185:
                raw_adj[h].add(t)
                raw_adj[t].add(h)
    filtered_adj = defaultdict(set)
    for h, t, _relation in side_data['entity_kg']['edge']:
        h, t = int(h), int(t)
        if h != t:
            filtered_adj[h].add(t)
            filtered_adj[t].add(h)
    return {'R0_filtered': filtered_adj, 'R1_raw': raw_adj}


def build_index(adjacency, movie_ids, max_movies_per_neighbor):
    """Inverted one-hop index: neighbor entity -> eligible movie entities.

    Shared-neighbor retrieval is two-hop: context -> neighbor -> movie.
    Hubs are excluded only for these shared-neighbor votes, not direct links.
    """
    movie_set = set(movie_ids)
    neighbor_to_movies = defaultdict(set)
    for movie in movie_ids:
        for neighbor in adjacency.get(movie, ()):
            neighbor_to_movies[neighbor].add(movie)
    neighbor_to_movies = {
        k: v for k, v in neighbor_to_movies.items()
        if 0 < len(v) <= max_movies_per_neighbor
    }
    return adjacency, movie_set, neighbor_to_movies


def retrieve(context_entities, index):
    adjacency, movie_set, neighbor_to_movies = index
    scores = Counter()
    for entity in set(context_entities):
        # Direct context->movie KG connections.
        for neighbor in adjacency.get(entity, ()):
            if neighbor in movie_set:
                scores[neighbor] += 1.0
        # Two-hop context->shared entity->movie paths.
        # IDF downweights attributes common to many eligible movies.
        for neighbor in adjacency.get(entity, ()):
            movies = neighbor_to_movies.get(neighbor)
            if movies:
                weight = 1.0 / math.log2(2.0 + len(movies))
                for movie in movies:
                    scores[movie] += weight
    # Deterministic ordering, independent of ground-truth label.
    return sorted(scores, key=lambda mid: (-scores[mid], mid))


def describe_result(counts, k_values):
    result = {'events': counts['events'], 'with_nonempty_context': counts['context'],
              'with_any_candidate': counts['retrieved'],
              'context_target_overlap_events': counts['overlap'],
              'mean_retrieved_candidates': counts['total_candidates'] / counts['events'] if counts['events'] else None,
              'mean_candidates_given_nonempty_retrieval': counts['total_candidates'] / counts['retrieved'] if counts['retrieved'] else None}
    for segment in ('overall', 'zero_shot', 'one_shot', 'other'):
        n = counts[f'{segment}_n']
        result[segment] = {'events': n}
        for k in k_values:
            result[segment][f'recall@{k}'] = counts[f'{segment}_hit@{k}'] / n if n else None
        result[segment]['candidate_recall_all'] = counts[f'{segment}_candidate_hit'] / n if n else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    parser.add_argument('--catalog', default='catalog_audit_corrected.pkl')
    parser.add_argument('--raw-dir', default='data/dataset/redial/nltk')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-movies-per-neighbor', type=int, default=500,
                        help='Fixed exploratory hub cutoff. Do not select using test.')
    parser.add_argument('--out', default='results_corrected/kg_retrieval_validation.json')
    args = parser.parse_args()

    import torch
    system, valid, _test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data['item_entity_ids'], system.device)
    official_movies = [int(x) for x in catalog['movie_ids']]
    freq_dict = {int(k): int(v) for k, v in catalog['meta']['freq_dict'].items()}
    graphs = load_graphs(side_data, Path(args.raw_dir))
    indexes = {name: build_index(adj, official_movies, args.max_movies_per_neighbor)
               for name, adj in graphs.items()}

    k_values = (50, 100, 500)
    counts = {name: Counter() for name in indexes}
    padding_id = int(valid.pad_entity_idx)
    n_events = 0
    # Follow exactly the recommender events used in M0/M1. No labels enter retrieve().
    with torch.no_grad():
        for batch in tqdm(valid.get_rec_data(batch_size=args.batch_size, shuffle=False), desc='KG retrieval validation'):
            ctx_batch, labels = batch[0], batch[-1]
            for context_row, label in zip(ctx_batch.tolist(), labels.tolist()):
                target = int(label)
                if target not in freq_dict:
                    raise ValueError(f'Target {target} absent from catalog metadata')
                context = [int(e) for e in context_row if int(e) != padding_id]
                freq = freq_dict[target]
                segment = 'zero_shot' if freq == 0 else ('one_shot' if freq == 1 else 'other')
                n_events += 1
                for name, index in indexes.items():
                    c = counts[name]
                    ranked = retrieve(context, index)
                    c['events'] += 1
                    c['overall_n'] += 1
                    c[f'{segment}_n'] += 1
                    c['context'] += bool(context)
                    c['retrieved'] += bool(ranked)
                    c['overlap'] += target in context
                    c['total_candidates'] += len(ranked)
                    if target in ranked:
                        c['overall_candidate_hit'] += 1
                        c[f'{segment}_candidate_hit'] += 1
                    for k in k_values:
                        if target in ranked[:k]:
                            c[f'overall_hit@{k}'] += 1
                            c[f'{segment}_hit@{k}'] += 1
    if n_events != 3010:
        raise RuntimeError(f'Expected 3010 validation recommendation events, observed {n_events}')

    report = {
        'split': 'validation_only',
        'notes': [
            'Target labels are used for scoring only, never candidate generation.',
            'Retrieval can return fewer than K items; no popularity padding or oracle completion.',
            'R0 and R1 share identical retrieval logic and fixed hub suppression.',
            'This is exploratory retrieval diagnostics, not M2 recommendation performance.',
            'A zero-shot item has zero training recommendation-target frequency, not necessarily zero training mentions.'
        ],
        'candidate_movies': len(official_movies),
        'padding_entity_id': padding_id,
        'hub_cutoff_movies': args.max_movies_per_neighbor,
        'results': {name: describe_result(c, k_values) for name, c in counts.items()}
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', target.resolve())


if __name__ == '__main__':
    main()
