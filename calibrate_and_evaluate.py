"""Consistent KGSF ranking/fairness evaluator and validation-selected calibration.

KGSF recommendation contract: (rec_loss, info_loss, rec_scores), batch[-1] is
an ENTITY ID. Topk over movie columns returns LOCAL movie indices.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system


EXPERIMENTS = [
    {'name': 'Baseline', 'lam': 0.0, 'gamma': 0.5, 'beta': 0.0},
    {'name': 'Conservative', 'lam': 0.8, 'gamma': 0.5, 'beta': 0.3},
    {'name': 'Balanced', 'lam': 1.4, 'gamma': 0.5, 'beta': 0.6},
    {'name': 'Aggressive', 'lam': 2.2, 'gamma': 0.5, 'beta': 1.0},
]


def gini(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or (a < 0).any():
        raise ValueError('Invalid exposure vector')
    if a.sum() == 0:
        return 0.0
    a.sort()
    n = len(a)
    return float(((2 * np.arange(1, n + 1) - n - 1) * a).sum() / (n * a.sum()))


def initialize(config_path, gpu):
    if not Path(config_path).is_file():
        raise FileNotFoundError(config_path)
    config = Config(config_path, gpu=gpu)
    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    vocab = dataset.vocab
    side_data = dataset.side_data
    train = get_dataloader(config, dataset.train_data, vocab)
    valid = get_dataloader(config, dataset.valid_data, vocab)
    test = get_dataloader(config, dataset.test_data, vocab)
    if not Path('save/KGSF.pth').exists():
        raise FileNotFoundError('save/KGSF.pth not found; preserve or restore your trained checkpoint')
    system = get_system(config, train, valid, test, vocab, side_data, restore_system=True)
    return system, valid, test, side_data


def load_catalog(path, official_movie_ids, device):
    with open(path, 'rb') as f:
        meta = pickle.load(f)  # only load locally generated, trusted pickle files
    movie_ids = [int(x) for x in official_movie_ids]
    if movie_ids != [int(x) for x in meta['movie_ids']]:
        raise ValueError('Catalog ordering differs from current KGSF item_entity_ids. Regenerate catalog.')
    groups = meta['item2group']
    frequencies = meta['freq_dict']
    if any(item not in groups or item not in frequencies for item in movie_ids):
        raise ValueError('Catalog contains movie(s) without groups/frequencies')
    freq = np.asarray([frequencies[item] for item in movie_ids], dtype=np.float32)
    denom = max(1.0, float(freq.max()))
    return {
        'meta': meta, 'movie_ids': movie_ids, 'candidate_ids': torch.tensor(movie_ids, device=device, dtype=torch.long),
        'position': {item: j for j, item in enumerate(movie_ids)},
        'norm_freq': torch.tensor(freq / denom, device=device, dtype=torch.float32),
        'g0_mask': torch.tensor([int(groups[item] == 0) for item in movie_ids], device=device, dtype=torch.float32),
        'group_ids': np.asarray([groups[item] for item in movie_ids], dtype=np.int64),
        'frequencies': freq,
    }


def evaluate(system, dataloader, catalog, settings, batch_size=32, k_values=(1, 10, 50)):
    """Evaluate one config on recommendation EVENTS (KGSF get_rec_data samples)."""
    system.model.eval()
    ids, pos = catalog['movie_ids'], catalog['position']
    max_k = max(k_values)
    if max_k > len(ids):
        raise ValueError('K exceeds eligible movie count')
    group_counts = np.zeros(5, dtype=np.int64)
    item_counts = np.zeros(len(ids), dtype=np.int64)
    hits = {k: 0 for k in k_values}
    ndcg = {k: 0.0 for k in k_values}
    mrr = {k: 0.0 for k in k_values}
    n_events = 0
    missing_targets = 0
    zero_events, one_events, low_events = 0, 0, 0
    zero_hits, one_hits, low_hits = 0, 0, 0
    k_exposure = min(10, max_k)
    lam, beta, gamma = settings['lam'], settings['beta'], settings['gamma']
    penalty = lam * catalog['norm_freq'].pow(gamma) - beta * catalog['g0_mask']
    with torch.no_grad():
        for batch in tqdm(dataloader.get_rec_data(batch_size=batch_size, shuffle=False), desc=settings['name']):
            batch = tuple(x.to(system.device) for x in batch)
            targets = batch[-1]
            output = system.model.recommend(batch, mode='test')
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError('Expected (rec_loss, info_loss, rec_scores) from KGSF.recommend()')
            scores = output[2]
            if scores.ndim != 2 or scores.size(0) != len(targets):
                raise RuntimeError(f'Invalid score shape {tuple(scores.shape)}')
            if max(ids) >= scores.size(1):
                raise RuntimeError('Movie entity ID beyond model score dimension')
            candidate_scores = scores.index_select(1, catalog['candidate_ids'])
            candidate_scores = candidate_scores - penalty.unsqueeze(0)
            topk = torch.topk(candidate_scores, max_k, dim=1).indices.cpu().numpy()
            target_ids = targets.cpu().tolist()
            for row, true_id in zip(topk, target_ids):
                n_events += 1
                true_id = int(true_id)
                local_target = pos.get(true_id)
                if local_target is None:
                    missing_targets += 1
                    continue  # diagnosed below, never silently drop from denominator
                rank_indices = np.flatnonzero(row == local_target)
                rank = int(rank_indices[0]) + 1 if len(rank_indices) else None
                for k in k_values:
                    if rank is not None and rank <= k:
                        hits[k] += 1
                        ndcg[k] += 1.0 / np.log2(rank + 1)
                        mrr[k] += 1.0 / rank
                true_f = catalog['meta']['freq_dict'][true_id]
                if true_f == 0:
                    zero_events += 1
                    zero_hits += int(rank is not None and rank <= 10)
                if true_f == 1:
                    one_events += 1
                    one_hits += int(rank is not None and rank <= 10)
                if true_f <= 1:
                    low_events += 1
                    low_hits += int(rank is not None and rank <= 10)
                exposed = row[:k_exposure]
                np.add.at(item_counts, exposed, 1)
                np.add.at(group_counts, catalog['group_ids'][exposed], 1)
    if n_events == 0:
        raise RuntimeError('No recommendation events were returned')
    if missing_targets:
        raise RuntimeError(f'{missing_targets}/{n_events} labels not in official movie catalog; inspect mapping')
    total_slots = int(item_counts.sum())
    assert total_slots == n_events * k_exposure
    assert int(group_counts.sum()) == total_slots
    shares = group_counts / total_slots
    return {
        'name': settings['name'], 'parameters': dict(settings),
        'n_recommendation_events': n_events, 'n_exposure_slots': total_slots,
        'hit': {str(k): hits[k] / n_events for k in k_values},
        'ndcg': {str(k): ndcg[k] / n_events for k in k_values},
        'mrr': {str(k): mrr[k] / n_events for k in k_values},
        'group_counts': group_counts.tolist(), 'group_shares': shares.tolist(),
        'item_gini': gini(item_counts), 'group_gini': gini(group_counts),
        'catalog_coverage_10': float(np.count_nonzero(item_counts) / len(ids)),
        'zero_shot_hit10': (zero_hits / zero_events) if zero_events else None,
        'one_shot_hit10': (one_hits / one_events) if one_events else None,
        'low_interaction_hit10': (low_hits / low_events) if low_events else None,
        'zero_shot_test_events': zero_events, 'one_shot_test_events': one_events,
        'low_interaction_test_events': low_events,
        'item_exposure_counts': {str(item_id): int(count) for item_id, count in zip(ids, item_counts)},
    }


def compact(result):
    return {key: value for key, value in result.items() if key != 'item_exposure_counts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    parser.add_argument('--catalog', default='catalog_audit_corrected.pkl')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out', default='results_corrected')
    parser.add_argument('--min-ndcg-retention', type=float, default=0.95,
                        help='Validation NDCG@10 must be at least this fraction of baseline')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    system, valid, test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data['item_entity_ids'], system.device)
    valid_results = [evaluate(system, valid, catalog, exp, args.batch_size) for exp in EXPERIMENTS]
    baseline_ndcg = valid_results[0]['ndcg']['10']
    eligible = [r for r in valid_results if r['ndcg']['10'] >= args.min_ndcg_retention * baseline_ndcg]
    if not eligible:
        raise RuntimeError('No calibration configuration meets validation utility constraint')
    # Explicit criterion: among valid configs, minimize item Gini; break ties by NDCG and name.
    selected = sorted(eligible, key=lambda r: (r['item_gini'], -r['ndcg']['10'], r['name']))[0]
    chosen = next(exp for exp in EXPERIMENTS if exp['name'] == selected['name'])
    print(f"Selected from VALIDATION: {chosen['name']} (item Gini={selected['item_gini']:.4f})")
    test_baseline = evaluate(system, test, catalog, EXPERIMENTS[0], args.batch_size)
    test_selected = test_baseline if chosen['name'] == 'Baseline' else evaluate(system, test, catalog, chosen, args.batch_size)
    payload = {
        'selection_criterion': f'minimize validation item_gini with NDCG@10 >= {args.min_ndcg_retention} * baseline',
        'selected_configuration': chosen,
        'validation': [compact(r) for r in valid_results],
        'test_baseline': compact(test_baseline), 'test_selected': compact(test_selected),
        'note': 'Test results are not used for hyperparameter selection; 10-slot exposure is unweighted.',
    }
    (out / 'calibration_results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    for filename, result in [('test_baseline_exposure.json', test_baseline), ('test_selected_exposure.json', test_selected)]:
        (out / filename).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    print(f'Saved results to {out.resolve()}')


if __name__ == '__main__':
    main()
