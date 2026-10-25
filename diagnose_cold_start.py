"""Phase 4: read-only KGSF cold-start rank, score, exposure and KG coverage audit.

Run from the CRSLab project root after finishing M0/M1:
    python diagnose_cold_start.py

The script does not train, update weights, or select new hyperparameters.
It uses the existing official movie candidate universe and checkpoint.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog, EXPERIMENTS


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {'n': int(a.size), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90))}


def kg_degree_summary(edges, n_entity, candidate_ids, target_ids):
    """Count non-self-loop *unique neighbors* in the processed DBpedia graph.

    This measures graph connectivity, not completeness or factual reliability.
    """
    neighbors = {int(i): set() for i in candidate_ids}
    for row in edges:
        if len(row) < 2:
            raise RuntimeError('Unexpected entity_kg edge format; expected (head, tail, relation)')
        h, t = int(row[0]), int(row[1])
        if h != t and h in neighbors:
            neighbors[h].add(t)
        if h != t and t in neighbors:
            neighbors[t].add(h)
    degree = {i: len(nb) for i, nb in neighbors.items()}
    return {
        'all_eligible_movies': summarize(list(degree.values())),
        'zero_shot_eligible_movies': summarize([degree[i] for i in candidate_ids if i in target_ids['zero_catalog']]),
        'one_shot_eligible_movies': summarize([degree[i] for i in candidate_ids if i in target_ids['one_catalog']]),
        'zero_shot_test_target_movies': summarize([degree[i] for i in target_ids['zero_test']]),
        'zero_shot_test_targets_no_nonself_neighbors': sum(degree[i] == 0 for i in target_ids['zero_test']),
        'zero_shot_catalog_no_nonself_neighbors': sum(degree[i] == 0 for i in target_ids['zero_catalog']),
        'notes': 'Unique processed-graph neighbors, excluding self-loops. Does not verify KG attributes or data quality.',
    }


def diagnose(system, dataloader, catalog, graph_edges, batch_size, selected):
    model, device = system.model, system.device
    model.eval()
    ids = np.asarray(catalog['movie_ids'], dtype=np.int64)
    freqs = np.asarray(catalog['frequencies'], dtype=np.float32)
    groups = np.asarray(catalog['group_ids'], dtype=np.int64)
    id_to_position = catalog['position']
    counts = {name: Counter() for name in ['baseline', 'selected']}
    target_rank = {name: {'zero': [], 'one': [], 'other': []} for name in counts}
    target_score = {name: {'zero': [], 'one': [], 'other': []} for name in counts}
    margins_top10 = {name: {'zero': [], 'one': [], 'other': []} for name in counts}
    correct_at10 = {name: {'zero': 0, 'one': 0, 'other': 0} for name in counts}
    total_events, missing_targets = 0, 0
    target_events = Counter()
    target_movie_ids = {'zero_test': set(), 'zero_catalog': set(ids[freqs == 0].tolist()),
                        'one_catalog': set(ids[freqs == 1].tolist())}
    # Calibration is frozen from validation. No parameter selection is performed here.
    penalty = selected['lam'] * catalog['norm_freq'].pow(selected['gamma']) - selected['beta'] * catalog['g0_mask']
    candidate_ids = catalog['candidate_ids']
    with torch.inference_mode():
        for batch in tqdm(dataloader.get_rec_data(batch_size=batch_size, shuffle=False), desc='Cold-start diagnosis'):
            batch = tuple(x.to(device) for x in batch)
            targets = batch[-1].detach().cpu().numpy().reshape(-1)
            outputs = model.recommend(batch, mode='test')
            if not isinstance(outputs, tuple) or len(outputs) != 3:
                raise RuntimeError('KGSF output contract changed: expected (rec_loss, info_loss, rec_scores)')
            raw = outputs[2]
            if raw.ndim != 2 or raw.shape[0] != len(targets):
                raise RuntimeError(f'Unexpected score shape {tuple(raw.shape)}')
            if max(ids) >= raw.shape[1]:
                raise RuntimeError('A movie entity ID is outside the model score dimension')
            scores = raw.index_select(1, candidate_ids)
            for name, matrix in [('baseline', scores), ('selected', scores - penalty.unsqueeze(0))]:
                # Exact rank, with strict-greater comparison. Equal scores are broken by item position.
                # Tie rate is reported so rank precision can be assessed.
                top10 = torch.topk(matrix, k=10, dim=1).indices.detach().cpu().numpy()
                counts[name].update(ids[top10.reshape(-1)].tolist())
                matrix_np = matrix.detach().cpu().numpy()
                cutoffs = np.partition(matrix_np, -10, axis=1)[:, -10]
                for j, target in enumerate(targets):
                    target = int(target)
                    local = id_to_position.get(target)
                    if local is None:
                        continue
                    group = 'zero' if freqs[local] == 0 else 'one' if freqs[local] == 1 else 'other'
                    tscore = float(matrix_np[j, local])
                    # Exact position within a deterministically defined total order:
                    # score descending, then local movie position ascending.
                    rank = 1 + int(np.count_nonzero(matrix_np[j] > tscore)) + int(np.count_nonzero((matrix_np[j] == tscore) & (np.arange(len(ids)) < local)))
                    target_rank[name][group].append(rank)
                    target_score[name][group].append(tscore)
                    margins_top10[name][group].append(tscore - float(cutoffs[j]))
                    correct_at10[name][group] += int(local in top10[j])
            for target in targets:
                total_events += 1
                local = id_to_position.get(int(target))
                if local is None:
                    missing_targets += 1
                else:
                    group = 'zero' if freqs[local] == 0 else 'one' if freqs[local] == 1 else 'other'
                    target_events[group] += 1
                    if group == 'zero':
                        target_movie_ids['zero_test'].add(int(target))
    if missing_targets:
        raise RuntimeError(f'{missing_targets} test targets absent from official movie catalog')
    if not total_events:
        raise RuntimeError('No recommendation events to diagnose')
    output = {
        'scope': 'Read-only diagnostic on held-out test data; no hyperparameter tuning',
        'recommendation_events': total_events,
        'target_events': dict(target_events),
        'selected_calibration_from_validation': dict(selected),
        'candidate_movies': int(len(ids)),
        'ranking': {},
        'kg_connectivity': kg_degree_summary(graph_edges, None, ids.tolist(), target_movie_ids),
    }
    for name in counts:
        exposed = counts[name]
        slot_total = sum(exposed.values())
        if slot_total != total_events * 10:
            raise RuntimeError('Incomplete top-10 exposure accounting')
        group_slots = [0] * 5
        zero_slots = one_slots = 0
        for movie_id, n in exposed.items():
            j = id_to_position[movie_id]
            group_slots[int(groups[j])] += n
            zero_slots += n if freqs[j] == 0 else 0
            one_slots += n if freqs[j] == 1 else 0
        if sum(group_slots) != slot_total:
            raise RuntimeError('Group exposure accounting mismatch')
        detail = {
            'exposure_slots': slot_total,
            'zero_shot_exposure_slots': int(zero_slots),
            'zero_shot_exposure_pct': 100 * zero_slots / slot_total,
            'one_shot_exposure_slots': int(one_slots),
            'one_shot_exposure_pct': 100 * one_slots / slot_total,
            'group_exposure_slots': group_slots,
            'distinct_exposed_zero_shot_movies': sum(freqs[id_to_position[i]] == 0 for i in exposed),
            'target_groups': {},
        }
        for group in ['zero', 'one', 'other']:
            ranks = target_rank[name][group]
            detail['target_groups'][group] = {
                'rank': summarize(ranks),
                'target_score': summarize(target_score[name][group]),
                'target_score_minus_top10_cutoff': summarize(margins_top10[name][group]),
                'hit_at_10': (correct_at10[name][group] / len(ranks)) if ranks else None,
                'hit_at_50': (sum(r <= 50 for r in ranks) / len(ranks)) if ranks else None,
                'fraction_rank_at_most_100': (sum(r <= 100 for r in ranks) / len(ranks)) if ranks else None,
            }
        output['ranking'][name] = detail
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    ap.add_argument('--catalog', default='catalog_audit_corrected.pkl')
    ap.add_argument('--calibration-results', default='results_corrected/calibration_results.json')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--out', default='results_corrected/cold_start_diagnosis.json')
    args = ap.parse_args()
    selection = json.loads(Path(args.calibration_results).read_text(encoding='utf-8'))
    selected = selection['selected_configuration']
    if not {'lam', 'gamma', 'beta'}.issubset(selected):
        raise ValueError('Selected calibration parameters missing; evaluate M1 first')
    system, valid, test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data['item_entity_ids'], system.device)
    report = diagnose(system, test, catalog, side_data['entity_kg']['edge'], args.batch_size, selected)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
    json.dumps(
        report,
        indent=2,
        default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj)
    ),
    encoding="utf-8")
    print('\nPHASE 4 — COLD-START DIAGNOSIS')
    print('Events:', report['recommendation_events'], '| candidate movies:', report['candidate_movies'])
    for name, result in report['ranking'].items():
        print('\n', name.upper())
        print('Zero-shot top-10 exposure:', result['zero_shot_exposure_slots'],
              f"({result['zero_shot_exposure_pct']:.4f}%)")
        print('One-shot top-10 exposure:', result['one_shot_exposure_slots'],
              f"({result['one_shot_exposure_pct']:.4f}%)")
        for group in ['zero', 'one', 'other']:
            g = result['target_groups'][group]
            print(f"  {group}: n={g['rank']['n']} median_rank={g['rank']['median']} "
                  f"Hit@10={g['hit_at_10']} Hit@50={g['hit_at_50']}")
    graph = report['kg_connectivity']
    print('\nZero-shot test target movie KG neighbors:', graph['zero_shot_test_target_movies'])
    print('Zero-shot test movies without non-self neighbors:', graph['zero_shot_test_targets_no_nonself_neighbors'])
    print('Saved:', target.resolve())


if __name__ == '__main__':
    main()
