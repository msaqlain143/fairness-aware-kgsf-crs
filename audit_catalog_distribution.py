import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from crslab.config import Config
from crslab.data import get_dataset

if __name__ == '__main__':
    print("=" * 60)
    print("Computing ReDial Catalog Interaction Frequencies & Groups")
    print("=" * 60)

    config_path = 'config/crs/kgsf/redial.yaml'
    config = Config(config_path)

    # Fast load from cached all_data.pkl
    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    
    train_data = dataset.train_data
    vocab = dataset.vocab
    
    freq_dict = {}
    total_interactions = 0

    for session in train_data:
        items = session.get('items', [])
        for item in items:
            freq_dict[item] = freq_dict.get(item, 0) + 1
            total_interactions += 1

    item_ids = np.array(list(freq_dict.keys()))
    frequencies = np.array(list(freq_dict.values()))

    # Sort ascending by interaction frequency
    sort_idx = np.argsort(frequencies)
    sorted_items = item_ids[sort_idx]
    sorted_freqs = frequencies[sort_idx]
    total_unique_items = len(sorted_items)

    # Rank-based Quintile Partitioning (Each quintile G_0 to G_4 gets exactly 20% of catalog items)
    group_assignments = np.zeros(total_unique_items, dtype=int)
    quintile_size = total_unique_items // 5

    for g in range(5):
        start = g * quintile_size
        end = (g + 1) * quintile_size if g < 4 else total_unique_items
        group_assignments[start:end] = g

    # Tau threshold bounds the highest interaction frequency inside G_0
    tau_nominal = sorted_freqs[group_assignments == 0].max()

    print(f"Total Unique Items Interacted in Train: {total_unique_items}")
    print(f"Total Item Interactions: {total_interactions}")
    print(f"Audit Threshold tau (Max freq in bottom quintile G_0): {tau_nominal}")

    print("\n--- Group Partitioning Statistics (Equal Catalog Quintiles) ---")
    group_meta = {}
    for g in range(5):
        mask = (group_assignments == g)
        count = np.sum(mask)
        pct = (count / total_unique_items) * 100
        min_f = sorted_freqs[mask].min()
        max_f = sorted_freqs[mask].max()
        group_meta[g] = {'count': count, 'min_freq': min_f, 'max_freq': max_f}
        print(f"Group G_{g}: {count} items ({pct:.1f}%) | Frequency Range: [{min_f}, {max_f}]")

    # Map each item_id directly to its assigned group {0..4}
    item2group = {item: grp for item, grp in zip(sorted_items, group_assignments)}

    # Save catalog partition for KGSF re-ranking calibration
    catalog_audit = {
        'freq_dict': freq_dict,
        'item2group': item2group,
        'sorted_items': sorted_items,
        'sorted_freqs': sorted_freqs,
        'tau': tau_nominal,
        'group_meta': group_meta
    }
    with open('catalog_audit.pkl', 'wb') as f:
        pickle.dump(catalog_audit, f)
    print("\n[Success] Saved catalog frequency metadata to 'catalog_audit.pkl'.")
    print("=" * 60)