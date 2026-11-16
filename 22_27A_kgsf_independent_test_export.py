#!/usr/bin/env python
"""Experiment 22.27A: frozen KGSF scores on the 4,003-event ReDial independent test set.

No training, retrieval rerun, or test-time parameter search is performed.
KGSF recommend() returns (rec_loss, info_loss, rec_scores); movie entity IDs
index columns of rec_scores.
"""
import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_record_conversation_ids(raw_test_path, processed_records,
                                         expected_raw_conversations=1342,
                                         expected_processed_records=14362):
    """Reproduce CRSLab role-merge/augmentation boundaries and verify roles."""
    raw_test_path = Path(raw_test_path)
    if not raw_test_path.is_file():
        raise FileNotFoundError(f"Raw ReDial test JSON not found: {raw_test_path}")
    with raw_test_path.open("r", encoding="utf-8") as f:
        raw_conversations = json.load(f)
    if not isinstance(raw_conversations, list):
        raise TypeError("Raw test JSON must contain a list of conversations.")
    if expected_raw_conversations is not None and len(raw_conversations) != expected_raw_conversations:
        raise RuntimeError(f"Expected {expected_raw_conversations} raw conversations; found {len(raw_conversations)}.")

    record_conv_ids, expected_roles, raw_conv_ids = [], [], []
    for conv_index, conversation in enumerate(raw_conversations):
        if "conv_id" not in conversation or "dialog" not in conversation:
            raise KeyError(f"Raw conversation {conv_index} lacks conv_id/dialog.")
        conv_id = int(conversation["conv_id"])
        if conv_id in raw_conv_ids:
            raise RuntimeError(f"Duplicate raw conversation ID: {conv_id}")
        raw_conv_ids.append(conv_id)
        dialog = conversation["dialog"]
        if not isinstance(dialog, list) or not dialog:
            raise ValueError(f"Conversation {conv_id} has an empty/invalid dialog.")
        merged_roles, last_role = [], None
        for utterance_index, utterance in enumerate(dialog):
            if "role" not in utterance:
                raise KeyError(f"Conversation {conv_id}, utterance {utterance_index} lacks role.")
            role = utterance["role"]
            if role != last_role:
                merged_roles.append(role)
            last_role = role
        roles_for_records = merged_roles[1:]
        record_conv_ids.extend([conv_id] * len(roles_for_records))
        expected_roles.extend(roles_for_records)

    if expected_processed_records is not None and len(processed_records) != expected_processed_records:
        raise RuntimeError(f"Expected {expected_processed_records} processed records; found {len(processed_records)}.")
    if len(record_conv_ids) != len(processed_records):
        raise RuntimeError(f"Raw/processed record count mismatch: {len(record_conv_ids)} vs {len(processed_records)}. Refusing positional alignment.")
    mismatches = []
    for i, (expected, record) in enumerate(zip(expected_roles, processed_records)):
        actual = record.get("role")
        if actual != expected:
            mismatches.append((i, expected, actual))
            if len(mismatches) == 10:
                break
    if mismatches:
        raise RuntimeError(f"Raw-to-processed role verification failed: {mismatches}")
    return record_conv_ids


def align_frozen_events(processed_records, record_conv_ids, mapping):
    required = {"event_id", "conv_id", "target_movie_id"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"Frozen mapping requires columns {sorted(required)}")
    if mapping[list(required)].isna().any().any():
        raise ValueError("Frozen event mapping contains missing values.")
    m = mapping.copy()
    for col in required:
        m[col] = pd.to_numeric(m[col], errors="raise").astype("int64")
    if m.event_id.duplicated().any():
        raise ValueError("Duplicate event_id in frozen mapping.")

    expected = Counter(zip(m.conv_id.tolist(), m.target_movie_id.tolist()))
    lookup = defaultdict(list)
    for row in m.sort_values("event_id").itertuples(index=False):
        lookup[(int(row.conv_id), int(row.target_movie_id))].append(int(row.event_id))
    observed = []
    for record_index, record in enumerate(processed_records):
        if record.get("role") != "Recommender":
            continue
        for key in ("items", "context_entities", "context_words"):
            if key not in record:
                raise KeyError(f"Recommender record {record_index} lacks {key}.")
        conv_id = int(record_conv_ids[record_index])
        for target in record["items"]:
            observed.append((conv_id, int(target), record_index))
    actual = Counter((cid, target) for cid, target, _ in observed)
    if expected != actual:
        missing, extra = expected - actual, actual - expected
        raise RuntimeError("Observed KGSF recommender events do not match frozen mapping. "
                           f"Expected={sum(expected.values())}, observed={sum(actual.values())}; "
                           f"missing={list(missing.items())[:8]}, extra={list(extra.items())[:8]}")
    used, events = Counter(), []
    for conv_id, target, record_index in observed:
        key = (conv_id, target)
        event_id = lookup[key][used[key]]
        used[key] += 1
        events.append({"event_id": event_id, "conv_id": conv_id,
                       "target_movie_id": target, "source_record_index": record_index})
    events.sort(key=lambda x: x["event_id"])
    if len(events) != len(m) or len({e["event_id"] for e in events}) != len(events):
        raise RuntimeError("Frozen event mapping is not one-to-one.")
    if set(m.conv_id) - set(record_conv_ids):
        raise RuntimeError("Some mapped conversations are absent from reconstructed records.")
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/crs/kgsf/redial.yaml")
    ap.add_argument("--checkpoint", default="save/KGSF.pth")
    ap.add_argument("--catalog", default="catalog_audit_corrected.pkl")
    ap.add_argument("--mapping", default="results_final_test_m2/m2_test_event_conversation_mapping.csv")
    ap.add_argument("--raw-test", default="data/dataset/redial/nltk/test_data.json")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--expected-events", type=int, default=4003)
    ap.add_argument("--expected-raw-conversations", type=int, default=1342)
    ap.add_argument("--expected-test-records", type=int, default=14362)
    ap.add_argument("--out-dir", default="results_final_test_m2/experiment_22_27A")
    args = ap.parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size and top-k must be positive.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.gpu.strip() == "-1" else args.gpu
    seed_all(args.seed)

    paths = [(Path(args.config), "KGSF config"), (Path(args.checkpoint), "KGSF checkpoint"),
             (Path(args.catalog), "frozen catalog"), (Path(args.mapping), "frozen event mapping"),
             (Path(args.raw_test), "raw independent-test JSON")]
    for path, label in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    print("=" * 76)
    print("EXPERIMENT 22.27A — KGSF INDEPENDENT-TEST EXPORT")
    print("=" * 76)
    config = Config(str(args.config), gpu=args.gpu)
    if str(config["dataset"]).lower() != "redial" or config["model_name"] != "KGSF":
        raise ValueError("This exporter requires the ReDial KGSF configuration.")
    dataset = get_dataset(config, config["tokenize"], restore=True, save=False)
    vocab, side_data = dataset.vocab, dataset.side_data
    train_loader = get_dataloader(config, dataset.train_data, vocab)
    valid_loader = get_dataloader(config, dataset.valid_data, vocab)
    test_loader = get_dataloader(config, dataset.test_data, vocab)
    system = get_system(config, train_loader, valid_loader, test_loader, vocab, side_data,
                        restore_system=False, interact=False, debug=False, tensorboard=False)
    device = system.device

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unexpected KGSF checkpoint structure: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
    system.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    system.model.eval()

    movie_ids = [int(x) for x in side_data["item_entity_ids"]]
    if len(movie_ids) != 6924 or len(set(movie_ids)) != 6924:
        raise RuntimeError(f"Expected 6,924 unique movie entity IDs; got {len(movie_ids)}.")
    with Path(args.catalog).open("rb") as f:
        catalog = pickle.load(f)  # trusted locally generated project artifact
    if movie_ids != [int(x) for x in catalog["movie_ids"]]:
        raise RuntimeError("KGSF movie ordering differs from frozen catalog ordering.")

    record_conv_ids = reconstruct_record_conversation_ids(
        args.raw_test, dataset.test_data, args.expected_raw_conversations, args.expected_test_records)
    events = align_frozen_events(dataset.test_data, record_conv_ids, pd.read_csv(args.mapping))
    if len(events) != args.expected_events:
        raise RuntimeError(f"Expected {args.expected_events} events; found {len(events)}.")

    # Construct native KGSF rec batches from each frozen event's original context.
    batch_records = []
    for event in events:
        rec = dataset.test_data[event["source_record_index"]]
        batch_records.append({"context_entities": rec["context_entities"],
                              "context_words": rec["context_words"],
                              "item": event["target_movie_id"]})

    n_events, n_movies = len(events), len(movie_ids)
    top_k = min(args.top_k, n_movies)
    score_matrix = np.empty((n_events, n_movies), dtype=np.float32)
    movie_index = torch.tensor(movie_ids, dtype=torch.long, device=device)
    print(f"Events={n_events} | conversations={len({e['conv_id'] for e in events})} | movies={n_movies}")
    print(f"Device={device} | batch_size={args.batch_size} | strict checkpoint load: PASS")

    with torch.no_grad():
        for start in tqdm(range(0, n_events, args.batch_size), desc="KGSF scoring"):
            stop = min(start + args.batch_size, n_events)
            native_batch = test_loader.rec_batchify(batch_records[start:stop])
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
                raise RuntimeError("KGSF produced invalid/non-finite movie scores.")
            score_matrix[start:stop] = movie_scores

    movie_position = {mid: i for i, mid in enumerate(movie_ids)}
    target_ranks, top_rows = [], []
    for i, event in enumerate(events):
        order = np.lexsort((np.asarray(movie_ids, dtype=np.int64), -score_matrix[i]))
        target = int(event["target_movie_id"])
        if target not in movie_position:
            raise RuntimeError(f"Target {target} is absent from official movie catalog.")
        rank = int(np.flatnonzero(order == movie_position[target])[0]) + 1
        target_ranks.append(rank)
        chosen = order[:top_k]
        top_rows.append({"event_id": event["event_id"], "conv_id": event["conv_id"],
                         "target_movie_id": target, "target_rank_full_catalog": rank,
                         "top_k": top_k,
                         "top_movie_ids": json.dumps([movie_ids[j] for j in chosen]),
                         "top_scores": json.dumps([float(score_matrix[i, j]) for j in chosen])})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "kgsf_independent_test_full_scores.npz"
    top_path = out_dir / "kgsf_independent_test_top500.csv"
    event_path = out_dir / "kgsf_independent_test_event_manifest.csv"
    summary_path = out_dir / "kgsf_independent_test_export_summary.json"
    validation_path = out_dir / "validation_checks.json"
    manifest_path = out_dir / "experiment_manifest.json"
    np.savez_compressed(scores_path, scores=score_matrix,
                        movie_ids=np.asarray(movie_ids, dtype=np.int64),
                        event_ids=np.asarray([e["event_id"] for e in events], dtype=np.int64))
    pd.DataFrame(top_rows).to_csv(top_path, index=False)
    event_manifest = pd.DataFrame(events)
    event_manifest["score_row"] = np.arange(n_events, dtype=np.int64)
    event_manifest["target_rank_full_catalog"] = target_ranks
    event_manifest.to_csv(event_path, index=False)

    checks = {
        "exactly_4003_events": len(events) == args.expected_events,
        "unique_event_ids": len({e["event_id"] for e in events}) == len(events),
        "exactly_1283_conversations": len({e["conv_id"] for e in events}) == 1283,
        "frozen_event_mapping_exact_multiset_match": True,
        "raw_to_processed_role_alignment": True,
        "strict_checkpoint_load": True,
        "official_catalog_order_matches": True,
        "catalog_has_6924_unique_movies": len(movie_ids) == 6924,
        "full_score_matrix_shape_valid": score_matrix.shape == (n_events, n_movies),
        "all_scores_finite": bool(np.isfinite(score_matrix).all()),
        "all_topk_rows_have_expected_length": all(len(json.loads(r["top_movie_ids"])) == top_k for r in top_rows),
        "targets_in_official_catalog": all(e["target_movie_id"] in movie_position for e in events),
        "training_performed": False,
        "parameter_search_performed": False,
        "test_labels_used_for_candidate_generation": False,
    }
    positive_checks = [v for k, v in checks.items() if isinstance(v, bool) and k not in
                       {"training_performed", "parameter_search_performed", "test_labels_used_for_candidate_generation"}]
    checks["overall_pass"] = all(positive_checks)
    if not checks["overall_pass"]:
        raise RuntimeError(f"Experiment validation failed: {checks}")
    validation_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    summary = {
        "experiment": "22_27A_kgsf_independent_test_export", "split": "independent_test",
        "events": n_events, "unique_conversations": len({e["conv_id"] for e in events}),
        "catalog_movies": n_movies, "score_matrix_shape": list(score_matrix.shape), "top_k": top_k,
        "target_rank_hit10_diagnostic": float(np.mean(np.asarray(target_ranks) <= 10)),
        "target_rank_hit50_diagnostic": float(np.mean(np.asarray(target_ranks) <= 50)),
        "test_split_evaluated": True, "training_performed": False, "parameter_search_performed": False,
        "files": {"full_scores": str(scores_path), "top_candidates": str(top_path),
                  "event_manifest": str(event_path), "validation": str(validation_path)}}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {**summary, "config": args.config, "checkpoint": args.checkpoint,
                "catalog": args.catalog, "event_mapping": args.mapping, "raw_test_json": args.raw_test,
                "checkpoint_load": "strict model_state_dict",
                "recommendation_interface": "system.model.recommend(batch, mode='test') -> (rec_loss, info_loss, rec_scores)",
                "score_columns": "official side_data['item_entity_ids'] order",
                "ranking": "descending KGSF score; movie entity ID ascending tie-break",
                "candidate_generation_uses_target": False, "training": False, "parameter_search": False}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n========== EXPERIMENT 22.27A COMPLETE ==========")
    print(f"Events: {n_events} | conversations: {summary['unique_conversations']} | score matrix: {score_matrix.shape}")
    print(f"Target-rank Hit@10 diagnostic: {summary['target_rank_hit10_diagnostic']:.6f}")
    print(f"Target-rank Hit@50 diagnostic: {summary['target_rank_hit50_diagnostic']:.6f}")
    print(json.dumps(checks, indent=2))
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEXPERIMENT 22.27A FAILED: {exc}", file=sys.stderr)
        raise



