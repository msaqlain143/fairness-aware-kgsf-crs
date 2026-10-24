"""Audit unchanged KGSF with the shared, movie-restricted evaluation protocol.

Uses calibrate_and_evaluate.py's evaluator so that M0 and M1 match exactly.
"""
import argparse
import json
from pathlib import Path

from calibrate_and_evaluate import initialize, load_catalog, evaluate, EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    parser.add_argument('--catalog', default='catalog_audit_corrected.pkl')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out', default='results_corrected/baseline_exposure.json')
    args = parser.parse_args()
    system, _, test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data['item_entity_ids'], system.device)
    result = evaluate(system, test, catalog, EXPERIMENTS[0], args.batch_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nBASELINE KGSF (movie-only top-10 exposure)')
    print(f"Recommendation events: {result['n_recommendation_events']}")
    print(f"Exposure slots: {result['n_exposure_slots']}")
    print(f"Hit@10: {result['hit']['10']:.4%}; Hit@50: {result['hit']['50']:.4%}")
    for group, (count, share) in enumerate(zip(result['group_counts'], result['group_shares'])):
        print(f'G{group}: {count:8d} slots, {share:.4%}')
    print(f"Item Gini: {result['item_gini']:.5f}; Group Gini: {result['group_gini']:.5f}")
    print(f"Zero-shot test events: {result['zero_shot_test_events']}")
    print(f'Saved {out.resolve()}')


if __name__ == '__main__':
    main()
