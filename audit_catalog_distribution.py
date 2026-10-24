"""Build train-only item popularity metadata for the official KGSF movie catalog.

Run from CRSLab repository root before fairness or calibration evaluation.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from crslab.config import Config
from crslab.data import get_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    parser.add_argument('--out', default='catalog_audit_corrected.pkl')
    args = parser.parse_args()
    if not Path(args.config).is_file():
        raise FileNotFoundError(f'Config not found: {args.config}')

    config = Config(args.config)
    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    raw_ids = [int(x) for x in dataset.side_data['item_entity_ids']]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError('Duplicate movie IDs in item_entity_ids; investigate catalog mapping')
    movie_ids = list(raw_ids)  # preserve official CRSLab movie-candidate ordering
    if not movie_ids:
        raise ValueError('Empty movie catalog')
    movie_set = set(movie_ids)

    # KGSF's rec_process_fn() creates one target per movie mentioned in
    # a Recommender-role utterance (see crslab/data/dataloader/kgsf.py).
    frequency = Counter()
    recommender_turns = 0
    target_occurrences = 0
    unknown_targets = Counter()
    for record in dataset.train_data:
        if record.get('role') != 'Recommender':
            continue
        recommender_turns += 1
        for value in record.get('items', []):
            item = int(value)
            target_occurrences += 1
            if item in movie_set:
                frequency[item] += 1
            else:
                unknown_targets[item] += 1
    if unknown_targets:
        raise ValueError(
            f'{sum(unknown_targets.values())} training recommendation targets do not belong '
            f'to the official movie catalog. Examples: {unknown_targets.most_common(5)}'
        )

    movie_freq = {item: int(frequency.get(item, 0)) for item in movie_ids}
    ranked = sorted(movie_ids, key=lambda item: (movie_freq[item], item))
    n = len(ranked)
    # Deterministic five groups, approximately equally sized; ties can be split.
    item2group = {item: min(5 * rank // n, 4) for rank, item in enumerate(ranked)}
    groups = {}
    for group in range(5):
        fs = [movie_freq[item] for item in movie_ids if item2group[item] == group]
        groups[group] = {'count': len(fs), 'min_freq': min(fs), 'max_freq': max(fs)}
        print(f'G{group}: {groups[group]}')
    zeros = sorted(item for item in movie_ids if movie_freq[item] == 0)
    ones = sorted(item for item in movie_ids if movie_freq[item] == 1)
    low_interaction = sorted(item for item in movie_ids if movie_freq[item] <= 1)
    meta = {
        'schema_version': 2,
        'movie_ids': movie_ids,
        'freq_dict': movie_freq,
        'item2group': item2group,
        'sorted_items': ranked,
        'zero_shot_ids': zeros,
        'one_shot_ids': ones,
        'low_interaction_ids': low_interaction,
        'group_meta': groups,
        'tau': 1,  # explicit low-interaction gate, NOT necessarily the G0 upper bound
        'g0_max_frequency': groups[0]['max_freq'],
        'frequency_definition': 'Number of movie targets in train Recommender-role utterances',
        'train_recommender_turns': recommender_turns,
        'train_target_occurrences': target_occurrences,
        'quintile_definition': 'Five rank-based catalog groups, sorted by (frequency, entity_id); frequency ties may cross boundaries',
    }
    out = Path(args.out)
    with out.open('wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        'eligible_movies': n, 'zero_shot_movies': len(zeros),
        'one_shot_movies': len(ones), 'train_recommender_turns': recommender_turns,
        'train_target_occurrences': target_occurrences,
        'group_counts': {str(g): groups[g] for g in groups},
        'g0_max_frequency': groups[0]['max_freq'], 'low_interaction_cutoff': 1,
    }
    Path('catalog_audit_corrected_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2))
    if sum(groups[g]['count'] for g in range(5)) != n:
        raise AssertionError('Group counts do not add up')
    print(f'Saved catalog metadata: {out}')


if __name__ == '__main__':
    main()
