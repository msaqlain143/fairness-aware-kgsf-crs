import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system

def move_batch_to_device(batch, device):
    """Recursively moves tensors in a batch dictionary to target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch

if __name__ == '__main__':
    print("=" * 60)
    print("Auditing Baseline KGSF Exposure Distribution Across Quintiles")
    print("=" * 60)

    # 1. Load Quintile Catalog Audit Data
    with open('catalog_audit.pkl', 'rb') as f:
        catalog_meta = pickle.load(f)
    item2group = catalog_meta['item2group']
    tau = catalog_meta['tau']

    config_path = 'config/crs/kgsf/redial.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), 'crslab', 'config', 'crs', 'kgsf', 'redial.yaml')
    
    config = Config(config_path, gpu='0')
    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    
    vocab = dataset.vocab
    side_data = dataset.side_data

    # 2. Build Dataloaders
    test_dataloader = get_dataloader(config, dataset.test_data, vocab)
    train_dataloader = get_dataloader(config, dataset.train_data, vocab)
    val_dataloader = get_dataloader(config, dataset.valid_data, vocab)

    # 3. Instantiate System with restore_system=True
    system = get_system(
        config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab,
        side_data,
        restore_system=True
    )

    print("\nRunning Top-K Inference over Test Set to Measure Exposure...")
    system.model.eval()
    
    K = 10
    group_recommendation_counts = {g: 0 for g in range(5)}
    total_recommended_items = 0
    device = system.device

    with torch.no_grad():
        for batch in tqdm(test_dataloader.get_rec_data(batch_size=32, shuffle=False)):
            batch = move_batch_to_device(batch, device)
            
            output = system.model.recommend(batch, mode='test')
            
            # Extract the actual 2D logits tensor [batch_size, n_items]
            if isinstance(output, tuple):
                scores = None
                for elem in output:
                    if isinstance(elem, torch.Tensor) and elem.dim() >= 2 and elem.size(-1) >= K:
                        scores = elem
                        break
                if scores is None:
                    # Fallback to the last element of the tuple
                    scores = output[-1]
            else:
                scores = output
            
            _, topk_indices = torch.topk(scores, k=K, dim=-1)
            topk_indices = topk_indices.cpu().numpy()

            for rec_list in topk_indices:
                for item_id in rec_list:
                    total_recommended_items += 1
                    grp = item2group.get(item_id, None)
                    if grp is not None:
                        group_recommendation_counts[grp] += 1

    print("\n" + "=" * 60)
    print("Baseline Popularity-Aware Exposure Distribution (Top-10):")
    print(f"Total Top-10 Recommendations Generated: {total_recommended_items}")
    print(f"Catalog Threshold Tau (G_0 Bound): {tau}")
    print("-" * 60)
    
    for g in range(5):
        count = group_recommendation_counts[g]
        pct = (count / total_recommended_items) * 100 if total_recommended_items > 0 else 0
        print(f"Group G_{g} (Catalog 20%): {count:6d} recommendations ({pct:5.2f}% exposure)")
    print("=" * 60)