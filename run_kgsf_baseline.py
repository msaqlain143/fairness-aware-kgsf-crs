"""Train and save the original KGSF recommendation backbone (no KG-RAG changes).

Run only if a new checkpoint is needed. Existing save/KGSF.pth must not be
silently replaced.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/crs/kgsf/redial.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--rec-epochs', type=int, default=3)
    parser.add_argument('--gpu', default='0', help="CRSLab GPU value; use '-1' for CPU if supported")
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing existing KGSF.pth')
    args = parser.parse_args()

    path = Path(args.config)
    if not path.is_file():
        raise FileNotFoundError(f'CRSLab config not found: {path.resolve()}')
    if args.batch_size <= 0 or args.pretrain_epochs < 0 or args.rec_epochs < 0:
        raise ValueError('Invalid batch size or epoch count')

    checkpoint = Path('save/KGSF.pth')
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f'{checkpoint} already exists. This script does not need to be rerun for the '
            'fairness fixes. Back it up and pass --overwrite ONLY if you intend to retrain.'
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = Config(str(path), gpu=args.gpu)
    config['pretrain']['batch_size'] = args.batch_size
    config['pretrain']['epoch'] = args.pretrain_epochs
    config['rec']['batch_size'] = args.batch_size
    config['rec']['epoch'] = args.rec_epochs
    print('Training original KGSF recommendation model; conversation training is excluded.')
    print(f'CUDA available: {torch.cuda.is_available()}')

    dataset = get_dataset(config, config['tokenize'], restore=True, save=False)
    vocab, side_data = dataset.vocab, dataset.side_data
    train_loader = get_dataloader(config, dataset.train_data, vocab)
    valid_loader = get_dataloader(config, dataset.valid_data, vocab)
    test_loader = get_dataloader(config, dataset.test_data, vocab)
    system = get_system(
        config, train_loader, valid_loader, test_loader, vocab, side_data,
        restore_system=False,
    )

    system.pretrain()
    system.train_recommender()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    system.save_model()
    if not checkpoint.exists():
        raise RuntimeError(f'Expected checkpoint not found after save_model(): {checkpoint}')
    record = {
        'config_path': str(path), 'seed': args.seed,
        'batch_size': args.batch_size, 'pretrain_epochs': args.pretrain_epochs,
        'rec_epochs': args.rec_epochs, 'torch_version': torch.__version__,
        'checkpoint': str(checkpoint), 'conv_training_performed': False,
    }
    record_path = checkpoint.parent / 'KGSF_training_metadata.json'
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Saved {checkpoint} and {record_path}')


if __name__ == '__main__':
    main()