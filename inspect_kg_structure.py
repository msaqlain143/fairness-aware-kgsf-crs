"""Phase 5.1: read-only raw-vs-KGSF-filtered DBpedia graph audit.

Run from the repository root: python inspect_kg_structure.py
Only reads the public dataset files; writes a JSON summary to results_corrected/.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'dataset' / 'redial' / 'nltk'
OUT = ROOT / 'results_corrected' / 'kg_structure_audit.json'
SELF_LOOP_ID = 185
THRESHOLD = 1000


def load_json(name):
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; run from your repository root')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def summarize(values):
    values = sorted(values)
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {'n': len(values), 'mean': sum(values)/len(values),
            'median': statistics.median(values),
            'p10': values[int((len(values)-1)*0.1)],
            'p90': values[int((len(values)-1)*0.9)]}


def main():
    entity2id = load_json('entity2id.json')
    raw = load_json('dbpedia_subkg.json')
    movies = {int(v) for v in load_json('movie_ids.json')}
    ids = {int(v) for v in entity2id.values()}
    n_entity = max(ids) + 1

    # Mirror ReDialDataset._entity_kg_process: a self-loop per source node;
    # two directed edges for every non-self raw triple (including duplicates).
    relation_counts = Counter()
    raw_nonself = defaultdict(set)
    invalid_tails = 0
    for key, pairs in raw.items():
        head = int(key)
        if head not in ids:
            continue
        relation_counts[SELF_LOOP_ID] += 1
        for rel, tail in pairs:
            rel, tail = int(rel), int(tail)
            if head == tail or rel == SELF_LOOP_ID:
                continue
            if tail not in ids:
                invalid_tails += 1
                continue
            relation_counts[rel] += 2
            raw_nonself[head].add(tail)
            raw_nonself[tail].add(head)

    kept_relations = {r for r, count in relation_counts.items() if count > THRESHOLD}
    filtered_nonself = defaultdict(set)
    filtered_edges = set()
    for key, pairs in raw.items():
        head = int(key)
        if head not in ids:
            continue
        if SELF_LOOP_ID in kept_relations:
            filtered_edges.add((head, head, SELF_LOOP_ID))
        for rel, tail in pairs:
            rel, tail = int(rel), int(tail)
            if head == tail or rel == SELF_LOOP_ID or tail not in ids or rel not in kept_relations:
                continue
            filtered_edges.add((head, tail, rel))
            filtered_edges.add((tail, head, rel))
            filtered_nonself[head].add(tail)
            filtered_nonself[tail].add(head)

    eligible = movies & ids
    raw_degree = {m: len(raw_nonself[m]) for m in eligible}
    filtered_degree = {m: len(filtered_nonself[m]) for m in eligible}
    gained = {m for m in eligible if raw_degree[m] > filtered_degree[m]}
    report = {
        'source': str(DATA),
        'filter_definition': 'original edge frequency > 1000, after reverse-edge augmentation; self relation 185',
        'entity_count': len(ids),
        'n_entity_index_range': n_entity,
        'raw_source_entities': len(raw),
        'movie_ids_in_file': len(movies),
        'eligible_movie_ids_found_in_entities': len(eligible),
        'invalid_raw_tail_links_skipped': invalid_tails,
        'raw_relation_types': len(relation_counts),
        'kept_relation_types': len(kept_relations),
        'raw_unique_undirected_nonself_pairs': len({(min(h,t),max(h,t)) for h, ts in raw_nonself.items() for t in ts}),
        'filtered_directed_edges_including_self': len(filtered_edges),
        'raw_movie_degree': summarize(list(raw_degree.values())),
        'filtered_movie_degree': summarize(list(filtered_degree.values())),
        'raw_movies_with_nonself_neighbors': sum(d>0 for d in raw_degree.values()),
        'filtered_movies_with_nonself_neighbors': sum(d>0 for d in filtered_degree.values()),
        'movies_gaining_neighbors_from_raw_kg': len(gained),
        'raw_only_nonself_neighbor_links_for_movies': sum(raw_degree[m]-filtered_degree[m] for m in eligible),
        'top_raw_relation_frequencies': [{'relation_id': int(r), 'augmented_edge_frequency': count, 'kept': r in kept_relations} for r,count in relation_counts.most_common(15)],
        'note': 'Coverage is descriptive, not evidence of relevance, zero-shot accuracy, or unseen-at-training status. Filter reconstruction assumes input format in redial.py.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PHASE 5.1 — RAW VS FILTERED KG')
    for k in ('entity_count','movie_ids_in_file','eligible_movie_ids_found_in_entities','raw_relation_types','kept_relation_types','raw_movies_with_nonself_neighbors','filtered_movies_with_nonself_neighbors','movies_gaining_neighbors_from_raw_kg','raw_only_nonself_neighbor_links_for_movies','invalid_raw_tail_links_skipped'):
        print(f'{k}: {report[k]}')
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
