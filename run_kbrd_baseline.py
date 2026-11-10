#!/usr/bin/env python
"""
Controlled recommendation-only KBRD training for KG-FairCRS Experiment 22.4A.

Uses CRSLab's existing ReDial dataset, dataloader, KBRD model/system and optimizer.
It deliberately does NOT call KBRDSystem.fit() or train_recommender(), because
the repository's standard methods proceed to test-set evaluation.

Training and model selection use train/validation only. The test dataloader is
constructed by CRSLab for API compatibility but is never iterated or evaluated.

Checkpoint format is CRSLab-compatible: {'model_state_dict': ...}.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_tensor_batch(batch, device):
    # Match BaseSystem.step(): move tensor values only. KBRD context_entities
    # may be a list of entity-ID lists and is handled by the model itself.
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/crs/kbrd/redial.yaml")
    parser.add_argument("--gpu", default="-1",
                        help="GPU ID, e.g. 0; use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Defaults to rec.epoch in YAML (10).")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--output", default="save/KBRD.pth")
    parser.add_argument("--metadata", default="save/KBRD_training_metadata.json")
    args = parser.parse_args()

    if args.patience < 1:
        raise ValueError("--patience must be >= 1.")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be >= 0.")

    # Set visible GPU before CRSLab constructs the model/device.
    if args.gpu.strip() == "-1":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from crslab.config import Config
    from crslab.data import get_dataset, get_dataloader
    from crslab.system import get_system

    set_seed(args.seed)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = Config(str(config_path), args.gpu)
    if str(config["dataset"]).lower() != "redial":
        raise ValueError(
            f"This experiment expects ReDial, got {config['dataset']!r}."
        )
    if config.get("model_name", "KBRD") != "KBRD":
        raise ValueError(
            "Expected model_name=KBRD in the resolved CRSLab config; "
            f"got {config.get('model_name')!r}."
        )

    # Use the same CRSLab dataset and dataloader construction as quick_start.
    dataset = get_dataset(config, config["tokenize"], False, False)
    train_loader = get_dataloader(config, dataset.train_data, dataset.vocab)
    valid_loader = get_dataloader(config, dataset.valid_data, dataset.vocab)
    test_loader = get_dataloader(config, dataset.test_data, dataset.vocab)

    system = get_system(
        config,
        train_loader,
        valid_loader,
        test_loader,
        dataset.vocab,
        dataset.side_data,
        restore_system=False,
        interact=False,
        debug=False,
        tensorboard=False,
    )

    # Explicit output path; preserve any existing KGSF checkpoint.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    system.model_file = str(output_path)

    epochs = int(args.epochs if args.epochs is not None
                 else system.rec_optim_opt["epoch"])
    batch_size = int(system.rec_batch_size)
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and recommendation batch size must be positive.")

    # CRSLab's own optimizer/backward implementation is retained.
    system.init_optim(system.rec_optim_opt, system.model.parameters())
    system.model.train()

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.time()

    print("=" * 78)
    print("EXPERIMENT 22.4A — CONTROLLED KBRD RECOMMENDATION TRAINING")
    print("=" * 78)
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {config['model_name']}")
    print(f"Device: {system.device}")
    print(f"Training epochs: {epochs} | batch size: {batch_size}")
    print(f"Early-stopping patience: {args.patience}")
    print(f"Checkpoint: {output_path.resolve()}")
    print("Test dataloader: constructed for CRSLab compatibility; NEVER iterated.")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        system.model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for batch in train_loader.get_rec_data(batch_size):
            batch = move_tensor_batch(batch, system.device)
            system.optimizer.zero_grad(set_to_none=True)

            loss, scores = system.model.forward(batch, mode="train", stage="rec")
            if not torch.isfinite(loss).all():
                raise RuntimeError(
                    f"Non-finite KBRD training loss at epoch {epoch}."
                )
            loss_sum = loss.sum()
            system.backward(loss_sum)

            n = int(batch["item"].shape[0])
            train_loss_sum += float(loss_sum.detach().cpu())
            train_examples += n

        if train_examples == 0:
            raise RuntimeError("KBRD training dataloader produced zero examples.")
        train_loss = train_loss_sum / train_examples

        # Validation only: no test loader access and no test metrics.
        system.model.eval()
        valid_loss_sum = 0.0
        valid_examples = 0
        with torch.no_grad():
            for batch in valid_loader.get_rec_data(batch_size, shuffle=False):
                batch = move_tensor_batch(batch, system.device)
                loss, scores = system.model.forward(
                    batch, mode="valid", stage="rec"
                )
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"Non-finite KBRD validation loss at epoch {epoch}."
                    )
                loss_sum = loss.sum()
                n = int(batch["item"].shape[0])
                valid_loss_sum += float(loss_sum.detach().cpu())
                valid_examples += n

        if valid_examples == 0:
            raise RuntimeError("KBRD validation dataloader produced zero examples.")
        valid_loss = valid_loss_sum / valid_examples

        improved = valid_loss < (best_val - args.min_delta)
        if improved:
            best_val = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in system.model.state_dict().items()
            }
            # Save immediately in a CRSLab-compatible checkpoint structure.
            torch.save({"model_state_dict": best_state}, output_path)
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "train_recommendation_loss": train_loss,
            "validation_recommendation_loss": valid_loss,
            "best_validation_loss_so_far": best_val,
            "improved": bool(improved),
            "seconds": time.time() - epoch_start,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.6f} | "
            f"valid loss={valid_loss:.6f} | "
            f"best epoch={best_epoch} | "
            f"no-improve={epochs_without_improvement}/{args.patience}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_epoch == 0 or not output_path.is_file():
        raise RuntimeError("No best validation checkpoint was produced.")

    # Restore the best validation-selected weights in memory and save using
    # CRSLab's native method, preserving its normal checkpoint conventions.
    system.model.load_state_dict(best_state)
    system.save_model()

    # Verify the saved checkpoint is readable and has the expected state dict.
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Saved checkpoint lacks model_state_dict.")
    if set(checkpoint["model_state_dict"]) != set(system.model.state_dict()):
        raise RuntimeError("Saved checkpoint keys do not match KBRD model.")

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "22.4A_KBRD_recommendation_only_training",
        "dataset": str(config["dataset"]),
        "model": str(config["model_name"]),
        "config": str(config_path),
        "seed": args.seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_recommendation_loss": best_val,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": batch_size,
        "checkpoint": str(output_path),
        "checkpoint_format": "CRSLab model_state_dict",
        "training_history": history,
        "test_loader_constructed": True,
        "test_loader_iterated": False,
        "test_evaluated": False,
        "elapsed_seconds": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("EXPERIMENT 22.4A COMPLETE")
    print("=" * 78)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation recommendation loss: {best_val:.8f}")
    print(f"Checkpoint: {output_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print("Checkpoint verification: PASS")
    print("Test split evaluated: NO")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEXPERIMENT 22.4A FAILED: {exc}", file=sys.stderr)
        raise
