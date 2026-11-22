#!/usr/bin/env python
"""Experiment 22.35A: export frozen KGSF validation scores and Top-K candidates.

Reuses the native CRSLab KGSF inference path and the existing validation event
manifest. It does not train, tune, modify checkpoints, or evaluate the independent
 test split. Event order is checked against valid.get_rec_data() before scoring.

Default input manifest:
  results_corrected/experiment_22_04B/kbrd_validation_event_manifest.csv

Default outputs:
  results_corrected/experiment_22_35/kgsf_validation/
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_record_conversation_ids(raw_valid_path: Path, processed_records) -> list[int]:
    """Reconstruct processed-record conversation IDs from raw validation roles.

    This mirrors the role-transition mapping used by the original 22.27A test
    exporter. It refuses positional alignment if raw and processed records do
    not agree exactly.
    """
    if not raw_valid_path.is_file():
        raise FileNotFoundError(f"Raw validation JSON not found: {raw_valid_path}")
    with raw_valid_path.open("r", encoding="utf-8") as fh:
        conversations = json.load(fh)
    if not isinstance(conversations, list):
        raise TypeError("Raw validation JSON must contain a list of conversations.")

    record_conv_ids: list[int] = []
    expected_roles: list[str] = []
    seen_conv_ids: set[int] = set()
    for conv_index, conversation in enumerate(conversations):
        if "conv_id" not in conversation or "dialog" not in conversation:
            raise KeyError(f"Raw validation conversation {conv_index} lacks conv_id/dialog.")
        conv_id = int(conversation["conv_id"])
        if conv_id in seen_conv_ids:
            raise RuntimeError(f"Duplicate raw validation conversation ID: {conv_id}")
        seen_conv_ids.add(conv_id)
        dialog = conversation["dialog"]
        if not isinstance(dialog, list) or not dialog:
            raise ValueError(f"Validation conversation {conv_id} has an empty/invalid dialog.")
        merged_roles, last_role = [], None
        for utterance_index, utterance in enumerate(dialog):
            if "role" not in utterance:
                raise KeyError(
                    f"Validation conversation {conv_id}, utterance {utterance_index} lacks role."
                )
            role = utterance["role"]
            if role != last_role:
                merged_roles.append(role)
            last_role = role
        roles_for_records = merged_roles[1:]
        record_conv_ids.extend([conv_id] * len(roles_for_records))
        expected_roles.extend(roles_for_records)

    if len(record_conv_ids) != len(processed_records):
        raise RuntimeError(
            "Raw/processed validation record count mismatch: "
            f"raw-derived={len(record_conv_ids)}, processed={len(processed_records)}. "
            "Refusing positional conversation alignment."
        )
    mismatches = [
        (i, expected_roles[i], processed_records[i].get("role"))
        for i in range(len(processed_records))
        if processed_records[i].get("role") != expected_roles[i]
    ][:10]
    if mismatches:
        raise RuntimeError(f"Raw-to-processed validation role verification failed: {mismatches}")
    return record_conv_ids


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Validation event manifest not found: {path}")
    df = pd.read_csv(path)
    required = {
        "event_id", "conv_id", "target_movie_id", "source_record_index",
        "score_row",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Validation manifest missing columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise RuntimeError("Validation manifest contains missing required values.")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    df = df.sort_values("event_id").reset_index(drop=True)
    if df.event_id.duplicated().any():
        raise RuntimeError("Duplicate event_id in validation manifest.")
    if df.conv_id.isna().any() or df.target_movie_id.isna().any():
        raise RuntimeError("Missing conversation or target IDs in validation manifest.")
    if not np.array_equal(df.score_row.to_numpy(), np.arange(len(df), dtype=np.int64)):
        raise RuntimeError("Manifest score_row must be contiguous 0..N-1 after event sorting.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/crs/kgsf/redial.yaml")
    ap.add_argument("--checkpoint", default="save/KGSF.pth")
    ap.add_argument("--catalog", default="catalog_audit_corrected.pkl")
    ap.add_argument(
        "--manifest",
        default="results_corrected/experiment_22_04B/kbrd_validation_event_manifest.csv",
    )
    ap.add_argument("--raw-valid", default="data/dataset/redial/nltk/valid_data.json")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--expected-events", type=int, default=3010)
    ap.add_argument("--expected-catalog-size", type=int, default=6924)
    ap.add_argument("--out-dir", default="results_corrected/experiment_22_35/kgsf_validation")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size and top-k must be positive")
    if args.expected_events < 1:
        raise ValueError("expected-events must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.gpu.strip() == "-1" else args.gpu
    seed_all(args.seed)

    config_path, checkpoint_path, catalog_path, manifest_path, raw_valid_path = map(
        Path, (args.config, args.checkpoint, args.catalog, args.manifest, args.raw_valid)
    )
    for path, label in (
        (config_path, "KGSF config"),
        (checkpoint_path, "frozen KGSF checkpoint"),
        (catalog_path, "frozen catalog"),
        (manifest_path, "validation event manifest"),
        (raw_valid_path, "raw validation JSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_names = [
        "kgsf_validation_full_scores.npz",
        "kgsf_validation_top500.csv",
        "kgsf_validation_event_manifest.csv",
        "kgsf_validation_export_summary.json",
        "validation_checks.json",
        "experiment_manifest.json",
    ]
    existing = [name for name in output_names if (out / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing 22.35A outputs: {existing}. "
            "Choose another --out-dir or pass --overwrite explicitly."
        )

    print("=" * 76)
    print("EXPERIMENT 22.35A — FROZEN KGSF VALIDATION EXPORT")
    print("=" * 76)
    manifest = load_manifest(manifest_path)
    if len(manifest) != args.expected_events:
        raise RuntimeError(
            f"Expected {args.expected_events} validation events; found {len(manifest)}"
        )
    print(f"Manifest events: {len(manifest)}")
    print(f"Unique conversations: {manifest.conv_id.nunique()}")

    config = Config(str(config_path), gpu=args.gpu)
    if str(config["dataset"]).lower() != "redial" or config["model_name"] != "KGSF":
        raise ValueError("This exporter requires the ReDial KGSF configuration.")

    dataset = get_dataset(config, config["tokenize"], restore=True, save=False)
    vocab, side_data = dataset.vocab, dataset.side_data
    train_loader = get_dataloader(config, dataset.train_data, vocab)
    valid_loader = get_dataloader(config, dataset.valid_data, vocab)
    test_loader = get_dataloader(config, dataset.test_data, vocab)
    system = get_system(
        config, train_loader, valid_loader, test_loader, vocab, side_data,
        restore_system=False, interact=False, debug=False, tensorboard=False,
    )
    device = system.device

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Unexpected KGSF checkpoint structure: "
            f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
        )
    system.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    system.model.eval()

    movie_ids = [int(x) for x in side_data["item_entity_ids"]]
    if len(movie_ids) != args.expected_catalog_size or len(set(movie_ids)) != len(movie_ids):
        raise RuntimeError(
            f"Expected {args.expected_catalog_size} unique catalog movie IDs; "
            f"got {len(movie_ids)}."
        )
    import pickle
    with catalog_path.open("rb") as fh:
        catalog = pickle.load(fh)  # trusted local project artifact
    catalog_movie_ids = [int(x) for x in catalog["movie_ids"]]
    if movie_ids != catalog_movie_ids:
        raise RuntimeError("KGSF item_entity_ids order differs from frozen catalog order.")

    # The existing manifest was built from validation recommendation events.
    # Verify the full target sequence against CRSLab's native validation iterator.
    dataloader_targets: list[int] = []
    for batch in valid_loader.get_rec_data(batch_size=args.batch_size, shuffle=False):
        dataloader_targets.extend(int(x) for x in batch[-1].tolist())
    manifest_targets = manifest.target_movie_id.astype("int64").tolist()
    if dataloader_targets != manifest_targets:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(dataloader_targets, manifest_targets)) if a != b),
            min(len(dataloader_targets), len(manifest_targets)),
        )
        raise RuntimeError(
            "Validation event target sequence does not match the frozen manifest: "
            f"dataloader={len(dataloader_targets)}, manifest={len(manifest_targets)}, "
            f"first_mismatch_index={mismatch}. Refusing positional alignment."
        )
    if len(dataloader_targets) != len(manifest):
        raise RuntimeError(
            f"Validation event count mismatch: loader={len(dataloader_targets)}, "
            f"manifest={len(manifest)}"
        )
    print("Native validation target sequence: PASS")

    # Resolve each manifest event to the exact processed validation record.
    valid_records = dataset.valid_data
    record_conv_ids = reconstruct_record_conversation_ids(raw_valid_path, valid_records)
    batch_records = []
    for row in manifest.itertuples(index=False):
        record_index = int(row.source_record_index)
        if record_index < 0 or record_index >= len(valid_records):
            raise RuntimeError(
                f"source_record_index out of range at event {row.event_id}: {record_index}"
            )
        rec = valid_records[record_index]
        if rec.get("role") != "Recommender":
            raise RuntimeError(
                f"Manifest event {row.event_id} points to non-Recommender record "
                f"{record_index}: role={rec.get('role')!r}"
            )
        for key in ("items", "context_entities", "context_words"):
            if key not in rec:
                raise KeyError(f"Validation record {record_index} lacks {key!r}.")
        if int(record_conv_ids[record_index]) != int(row.conv_id):
            raise RuntimeError(
                f"Manifest conversation {row.conv_id} does not match source record "
                f"{record_index} conversation {record_conv_ids[record_index]}"
            )
        record_targets = [int(x) for x in rec["items"]]
        if int(row.target_movie_id) not in record_targets:
            raise RuntimeError(
                f"Manifest target {row.target_movie_id} is not in record {record_index} items {record_targets}"
            )
        batch_records.append({
            "context_entities": rec["context_entities"],
            "context_words": rec["context_words"],
            "item": int(row.target_movie_id),
        })

    n_events, n_movies = len(manifest), len(movie_ids)
    top_k = min(args.top_k, n_movies)
    score_matrix = np.empty((n_events, n_movies), dtype=np.float32)
    movie_index = torch.tensor(movie_ids, dtype=torch.long, device=device)
    print(
        f"Events={n_events} | conversations={manifest.conv_id.nunique()} | "
        f"movies={n_movies} | device={device}"
    )

    with torch.no_grad():
        for start in tqdm(range(0, n_events, args.batch_size), desc="KGSF validation scoring"):
            stop = min(start + args.batch_size, n_events)
            native_batch = valid_loader.rec_batchify(batch_records[start:stop])
            native_batch = tuple(t.to(device) for t in native_batch)
            output = system.model.recommend(native_batch, mode="test")
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Expected KGSF recommend() -> (rec_loss, info_loss, rec_scores).")
            rec_scores = output[2]
            if rec_scores.ndim != 2 or rec_scores.shape[0] != stop - start:
                raise RuntimeError(f"Unexpected KGSF score shape: {tuple(rec_scores.shape)}")
            if max(movie_ids) >= rec_scores.shape[1]:
                raise RuntimeError("Movie entity ID exceeds KGSF score dimension.")
            movie_scores = rec_scores.index_select(1, movie_index).float().cpu().numpy()
            if movie_scores.shape != (stop - start, n_movies) or not np.isfinite(movie_scores).all():
                raise RuntimeError("KGSF produced invalid/non-finite validation movie scores.")
            score_matrix[start:stop] = movie_scores

    movie_position = {mid: i for i, mid in enumerate(movie_ids)}
    top_rows, target_ranks = [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        order = np.lexsort((np.asarray(movie_ids, dtype=np.int64), -score_matrix[i]))
        target = int(row.target_movie_id)
        if target not in movie_position:
            raise RuntimeError(f"Target {target} is absent from official movie catalog.")
        rank = int(np.flatnonzero(order == movie_position[target])[0]) + 1
        target_ranks.append(rank)
        chosen = order[:top_k]
        top_rows.append({
            "event_id": int(row.event_id),
            "conv_id": int(row.conv_id),
            "target_movie_id": target,
            "target_rank_full_catalog": rank,
            "top_k": top_k,
            "top_movie_ids": json.dumps([movie_ids[j] for j in chosen]),
            "top_scores": json.dumps([float(score_matrix[i, j]) for j in chosen]),
        })

    scores_path = out / "kgsf_validation_full_scores.npz"
    top_path = out / "kgsf_validation_top500.csv"
    event_path = out / "kgsf_validation_event_manifest.csv"
    summary_path = out / "kgsf_validation_export_summary.json"
    checks_path = out / "validation_checks.json"
    manifest_path_out = out / "experiment_manifest.json"

    event_manifest = manifest.copy()
    event_manifest["score_row"] = np.arange(n_events, dtype=np.int64)
    event_manifest["target_rank_full_catalog"] = target_ranks
    np.savez_compressed(
        scores_path,
        scores=score_matrix,
        movie_ids=np.asarray(movie_ids, dtype=np.int64),
        event_ids=manifest.event_id.to_numpy(dtype=np.int64),
        conv_ids=manifest.conv_id.to_numpy(dtype=np.int64),
        target_movie_ids=manifest.target_movie_id.to_numpy(dtype=np.int64),
    )
    pd.DataFrame(top_rows).to_csv(top_path, index=False)
    event_manifest.to_csv(event_path, index=False)

    checks = {
        "split": "validation",
        "events": n_events,
        "unique_event_ids": int(manifest.event_id.nunique()) == n_events,
        "unique_conversations": int(manifest.conv_id.nunique()),
        "manifest_target_sequence_matches_native_validation_loader": True,
        "manifest_source_record_indices_valid": True,
        "all_source_records_are_recommender": True,
        "raw_processed_validation_role_alignment": True,
        "manifest_conversation_matches_source_record": True,
        "all_manifest_targets_present_in_source_record_items": True,
        "strict_checkpoint_load": True,
        "official_catalog_order_matches": True,
        "catalog_movies": n_movies,
        "full_score_matrix_shape": list(score_matrix.shape),
        "all_scores_finite": bool(np.isfinite(score_matrix).all()),
        "top_k": top_k,
        "all_topk_rows_have_expected_length": all(
            len(json.loads(r["top_movie_ids"])) == top_k for r in top_rows
        ),
        "targets_in_official_catalog": all(int(x) in movie_position for x in manifest.target_movie_id),
        "training_performed": False,
        "parameter_search_performed": False,
        "independent_test_evaluated": False,
    }
    checks["overall_pass"] = all(
        value for key, value in checks.items()
        if isinstance(value, bool)
        and key not in {"training_performed", "parameter_search_performed", "independent_test_evaluated"}
    )
    if not checks["overall_pass"]:
        raise RuntimeError(f"22.35A validation failed: {checks}")
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    summary = {
        "experiment": "22.35A_kgsf_validation_export",
        "split": "validation",
        "events": n_events,
        "unique_conversations": int(manifest.conv_id.nunique()),
        "catalog_movies": n_movies,
        "score_matrix_shape": list(score_matrix.shape),
        "top_k": top_k,
        "target_rank_hit10_diagnostic": float(np.mean(np.asarray(target_ranks) <= 10)),
        "target_rank_hit50_diagnostic": float(np.mean(np.asarray(target_ranks) <= 50)),
        "training_performed": False,
        "parameter_search_performed": False,
        "independent_test_evaluated": False,
        "files": {
            "full_scores": str(scores_path),
            "top_candidates": str(top_path),
            "event_manifest": str(event_path),
            "validation_checks": str(checks_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    experiment_manifest = {
        **summary,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "catalog": str(catalog_path),
        "source_event_manifest": str(manifest_path),
        "raw_validation_json": str(raw_valid_path),
        "checkpoint_load": "strict model_state_dict",
        "recommendation_interface": "system.model.recommend(batch, mode='test') -> (rec_loss, info_loss, rec_scores)",
        "score_columns": "official side_data['item_entity_ids'] order",
        "ranking": "descending KGSF score; movie entity ID ascending tie-break",
        "candidate_generation_uses_target": False,
        "training": False,
        "parameter_search": False,
    }
    manifest_path_out.write_text(json.dumps(experiment_manifest, indent=2), encoding="utf-8")

    print("\n========== EXPERIMENT 22.35A COMPLETE ==========")
    print(f"Events: {n_events} | conversations: {manifest.conv_id.nunique()} | score matrix: {score_matrix.shape}")
    print(f"Target-rank Hit@10 diagnostic: {summary['target_rank_hit10_diagnostic']:.6f}")
    print(f"Target-rank Hit@50 diagnostic: {summary['target_rank_hit50_diagnostic']:.6f}")
    print(json.dumps(checks, indent=2))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
