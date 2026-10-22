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
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch

def compute_gini(allocations):
    arr = np.array(allocations, dtype=np.float64)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

def evaluate_calibration(system, test_dataloader, item2group, item_freq_tensor, g0_mask_tensor, lam=1.2, gamma=0.5, beta=0.4, K_list=[1, 10, 50]):
    system.model.eval()
    device = system.device
    
    hits = {k: 0 for k in K_list}
    ndcg = {k: 0.0 for k in K_list}
    mrr = {k: 0.0 for k in K_list}
    
    group_recommendation_counts = {g: 0 for g in range(5)}
    total_samples = 0
    total_recommendations = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader.get_rec_data(batch_size=32, shuffle=False), desc=f"Eval lam={lam}, beta={beta}"):
            batch = move_batch_to_device(batch, device)
            
            # Ground-truth targets extraction:
            # If batch is a tuple, the target item id tensor is the last element (batch[-1])
            if isinstance(batch, (tuple, list)):
                targets = batch[-1]
            elif isinstance(batch, dict):
                targets = batch.get('item', batch.get('movie', list(batch.values())[-1]))
            else:
                targets = batch

            output = system.model.recommend(batch, mode='test')
            
            # Extract the actual 2D prediction logits [batch_size, n_items]
            if isinstance(output, tuple):
                scores = None
                for elem in output:
                    if isinstance(elem, torch.Tensor) and elem.dim() >= 2 and elem.size(-1) >= max(K_list):
                        scores = elem
                        break
                if scores is None:
                    scores = output[-1]
            else:
                scores = output

            # Apply Knowledge-Grounded Calibration Penalty
            calibrated_scores = scores - lam * (item_freq_tensor ** gamma) + beta * g0_mask_tensor

            max_k = max(K_list)
            _, topk_indices = torch.topk(calibrated_scores, k=max_k, dim=-1)
            
            topk_cpu = topk_indices.cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            batch_sz = targets_cpu.shape[0]
            total_samples += batch_sz

            for idx in range(batch_sz):
                target = targets_cpu[idx]
                rec_items = topk_cpu[idx]

                # Tally Top-10 Exposure distribution
                for it in rec_items[:10]:
                    total_recommendations += 1
                    grp = item2group.get(it, None)
                    if grp is not None:
                        group_recommendation_counts[grp] += 1

                # Calculate Utility Metrics
                for k in K_list:
                    top_k_slice = rec_items[:k]
                    if target in top_k_slice:
                        hits[k] += 1
                        rank = np.where(top_k_slice == target)[0][0] + 1
                        mrr[k] += 1.0 / rank
                        ndcg[k] += 1.0 / np.log2(rank + 1)

    hit_rates = {k: hits[k] / total_samples for k in K_list}
    ndcg_scores = {k: ndcg[k] / total_samples for k in K_list}
    mrr_scores = {k: mrr[k] / total_samples for k in K_list}
    
    group_shares = {
        g: (group_recommendation_counts[g] / total_recommendations) * 100 
        for g in range(5)
    }
    gini = compute_gini(list(group_shares.values()))

    return hit_rates, ndcg_scores, mrr_scores, group_shares, gini

if __name__ == '__main__':
    print("=" * 65)
    print("Executing Popularity Calibration Audit on KGSF (ReDial)")
    print("=" * 65)

    with open('catalog_audit.pkl', 'rb') as f:
        catalog_meta = pickle.load(f)
    item2group = catalog_meta['item2group']
    
    item_frequencies = catalog_meta.get('item_frequencies', None)
    if item_frequencies is None:
        item_frequencies = {item: (grp + 1) for item, grp in item2group.items()}

    config_path = 'config/crs/kgsf/redial.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), 'crslab', 'config', 'crs', 'kgsf', 'redial.yaml')

    config = Config(config_path, gpu='0')
    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    vocab = dataset.vocab
    side_data = dataset.side_data

    test_dataloader = get_dataloader(config, dataset.test_data, vocab)
    train_dataloader = get_dataloader(config, dataset.train_data, vocab)
    val_dataloader = get_dataloader(config, dataset.valid_data, vocab)

    system = get_system(
        config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab,
        side_data,
        restore_system=True
    )
    device = system.device

    # Get total entity/movie vocabulary space
    n_items = len(vocab['movie2id']) if 'movie2id' in vocab else vocab['n_entity']
    
    freq_arr = np.zeros(n_items, dtype=np.float32)
    g0_mask = np.zeros(n_items, dtype=np.float32)

    for item_id, grp in item2group.items():
        if item_id < n_items:
            freq_arr[item_id] = float(item_frequencies.get(item_id, 1.0))
            if grp == 0:
                g0_mask[item_id] = 1.0

    max_f = np.max(freq_arr) if np.max(freq_arr) > 0 else 1.0
    norm_freq = freq_arr / max_f

    item_freq_tensor = torch.tensor(norm_freq, device=device)
    g0_mask_tensor = torch.tensor(g0_mask, device=device)

    experiments = [
        {"name": "Unconstrained Baseline", "lam": 0.0, "gamma": 1.0, "beta": 0.0},
        {"name": "Conservative Calibration", "lam": 0.8, "gamma": 0.5, "beta": 0.3},
        {"name": "Balanced Calibration",    "lam": 1.4, "gamma": 0.5, "beta": 0.6},
        {"name": "Aggressive Parity",        "lam": 2.2, "gamma": 0.5, "beta": 1.0},
    ]

    results_table = []

    for exp in experiments:
        print(f"\nRunning {exp['name']} (lambda={exp['lam']}, beta={exp['beta']})...")
        hit_rates, ndcg_scores, mrr_scores, group_shares, gini = evaluate_calibration(
            system=system,
            test_dataloader=test_dataloader,
            item2group=item2group,
            item_freq_tensor=item_freq_tensor,
            g0_mask_tensor=g0_mask_tensor,
            lam=exp['lam'],
            gamma=exp['gamma'],
            beta=exp['beta']
        )
        results_table.append({
            "name": exp['name'],
            "hit10": hit_rates[10],
            "hit50": hit_rates[50],
            "ndcg10": ndcg_scores[10],
            "g0": group_shares[0],
            "g4": group_shares[4],
            "gini": gini
        })

    print("\n" + "=" * 90)
    print(f"{'Configuration':<25} | {'Hit@10':<7} | {'Hit@50':<7} | {'NDCG@10':<8} | {'G_0 Exp%':<8} | {'G_4 Exp%':<8} | {'Gini':<6}")
    print("-" * 90)
    for res in results_table:
        print(f"{res['name']:<25} | {res['hit10']*100:5.2f}% | {res['hit50']*100:5.2f}% | {res['ndcg10']:7.4f}  | {res['g0']:7.2f}% | {res['g4']:7.2f}% | {res['gini']:5.3f}")
    print("=" * 90)