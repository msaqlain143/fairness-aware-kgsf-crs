import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system

if __name__ == '__main__':
    print("=" * 60)
    print("Training KGSF Recommender & Saving Checkpoint on RTX 3060")
    print("=" * 60)

    config_path = 'config/crs/kgsf/redial.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), 'crslab', 'config', 'crs', 'kgsf', 'redial.yaml')

    config = Config(config_path, gpu='0')

    # Keep batch size within 6GB VRAM budget
    if 'pretrain' in config:
        config['pretrain']['batch_size'] = 32
        config['pretrain']['epoch'] = 2
    if 'rec' in config:
        config['rec']['batch_size'] = 32
        config['rec']['epoch'] = 3

    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    vocab = dataset.vocab
    side_data = dataset.side_data

    print("Building dataloaders...")
    train_dataloader = get_dataloader(config, dataset.train_data, vocab)
    val_dataloader = get_dataloader(config, dataset.valid_data, vocab)
    test_dataloader = get_dataloader(config, dataset.test_data, vocab)

    print("Initializing KGSF system...")
    system = get_system(
        config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab,
        side_data,
        False  # restore_system=False
    )

    print("\n[Phase 1] Dual-KG Pre-training (MIM)...")
    system.pretrain()

    print("\n[Phase 2] Recommendation Head Training & Validation...")
    system.train_recommender()

    print("\n[Phase 3] Saving Model Weights to save/KGSF.pth...")
    os.makedirs('save', exist_ok=True)
    system.save_model()
    print("Model successfully saved to 'save/KGSF.pth'!")
    print("=" * 60)