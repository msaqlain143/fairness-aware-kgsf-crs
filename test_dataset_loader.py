import os
import torch
from crslab.config import Config
from crslab.data import get_dataset

if __name__ == '__main__':
    print("=" * 60)
    print("Testing ReDial Dataset Loader & Subgraph Processing")
    print("=" * 60)

    config_path = 'config/crs/kgsf/redial.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), 'crslab', 'config', 'crs', 'kgsf', 'redial.yaml')

    config = Config(config_path)

    if torch.cuda.is_available():
        config['gpu'] = [0]
        print(f"CUDA Active: {torch.cuda.get_device_name(0)}")
    else:
        config['gpu'] = [-1]
        print("CUDA not detected, falling back to CPU.")

    if 'pretrain' in config:
        config['pretrain']['batch_size'] = 32
    if 'rec' in config:
        config['rec']['batch_size'] = 32
    if 'conv' in config:
        config['conv']['batch_size'] = 16

    print("Processing ReDial raw files into tensors (generating all_data.pkl)...")
    
    # Set restore=False so it builds all_data.pkl from the extracted files,
    # and save=True so it caches the result for instant loading in future runs.
    dataset = get_dataset(config, config['tokenize'], restore=False, save=True)

    vocab = dataset.vocab
    side_data = dataset.side_data

    print("\n" + "=" * 60)
    print("[Dataset Statistics Verified]")
    print(f"Total vocabulary size: {vocab['vocab_size']}")
    print(f"Total catalog entities (DBpedia): {vocab['n_entity']}")
    print(f"Total vocabulary words (ConceptNet): {vocab['n_word']}")
    
    train_data = dataset.train_data
    val_data = dataset.valid_data
    test_data = dataset.test_data

    print(f"Train conversation sessions: {len(train_data)}")
    print(f"Validation conversation sessions: {len(val_data)}")
    print(f"Test conversation sessions: {len(test_data)}")
    print("=" * 60)
    print("Dataset loader initialized and cached successfully!")
    print("=" * 60)