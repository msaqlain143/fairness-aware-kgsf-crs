#!/usr/bin/env python
"""Experiment 22.4B: export KBRD scores on the frozen 3,010-event ReDial validation set.

Conversation IDs are reconstructed from the original raw validation JSON.
CRSLab's _merge_conv_data merges consecutive turns with the same role, and
_augment_and_add emits one processed record for every merged turn except the
first. This script reproduces that record-boundary logic and verifies every
reconstructed role against the restored processed validation records.

Loads save/KBRD.pth, aligns validation targets to the frozen event mapping by
(conv_id, target_movie_id), and exports the full 6,924-movie score matrix.
The test dataloader is constructed for CRSLab compatibility but never iterated.
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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_validation_conversation_ids(
    raw_valid_path,
    processed_valid,
    expected_raw_conversations=901,
    expected_processed_records=10256,
):
    """Reproduce CRSLab's merge/augmentation boundaries and attach conv_id.

    _merge_conv_data merges adjacent utterances having the same role.
    _augment_and_add creates one output record for each merged turn after
    the first. Thus each raw conversation contributes
    len(consecutive-role-groups) - 1 processed records.

    Every reconstructed role is checked against the corresponding restored
    CRSLab record. No positional assumption is accepted without this check.
    """
    raw_valid_path = Path(raw_valid_path)
    if not raw_valid_path.is_file():
        raise FileNotFoundError(f"Raw ReDial validation JSON not found: {raw_valid_path}")

    with raw_valid_path.open("r", encoding="utf-8") as f:
        raw_conversations = json.load(f)

    if not isinstance(raw_conversations, list):
        raise TypeError(
            f"Expected a list in {raw_valid_path}, got {type(raw_conversations)}"
        )

    if expected_raw_conversations is not None and len(raw_conversations) != expected_raw_conversations:
        raise RuntimeError(
            f"Expected {expected_raw_conversations} raw validation conversations, "
            f"found {len(raw_conversations)}. Check dataset/split provenance."
        )

    record_conv_ids = []
    expected_roles = []
    raw_conv_ids = []

    for conv_index, conversation in enumerate(raw_conversations):
        if "conv_id" not in conversation or "dialog" not in conversation:
            raise KeyError(
                f"Raw conversation {conv_index} must contain conv_id and dialog; "
                f"keys={sorted(conversation.keys())}"
            )

        try:
            conv_id = int(conversation["conv_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Raw conversation {conv_index} has non-integer conv_id "
                f"{conversation['conv_id']!r}"
            ) from exc

        if conv_id in raw_conv_ids:
            raise RuntimeError(f"Duplicate raw conversation ID: {conv_id}")
        raw_conv_ids.append(conv_id)

        dialog = conversation["dialog"]
        if not isinstance(dialog, list) or not dialog:
            raise ValueError(f"Conversation {conv_id} has an empty/invalid dialog.")

        # Exact role-boundary equivalent of _merge_conv_data().
        merged_roles = []
        last_role = None
        for utterance_index, utterance in enumerate(dialog):
            if "role" not in utterance:
                raise KeyError(
                    f"Conversation {conv_id}, utterance {utterance_index} "
                    "has no role field."
                )
            role = utterance["role"]
            if role != last_role:
                merged_roles.append(role)
            last_role = role

        # Exact record-count/role equivalent of _augment_and_add():
        # no record is emitted for the first merged turn.
        conversation_record_roles = merged_roles[1:]
        record_conv_ids.extend([conv_id] * len(conversation_record_roles))
        expected_roles.extend(conversation_record_roles)

    if expected_processed_records is not None and len(processed_valid) != expected_processed_records:
        raise RuntimeError(
            f"Expected {expected_processed_records} restored validation records, "
            f"found {len(processed_valid)}."
        )

    if len(record_conv_ids) != len(processed_valid):
        raise RuntimeError(
            "Reconstructed raw-to-processed record count mismatch: "
            f"raw reconstruction={len(record_conv_ids)}, "
            f"CRSLab restored records={len(processed_valid)}. "
            "Do not proceed with positional alignment."
        )

    # Verify the role sequence record-by-record, not just the total count.
    role_mismatches = []
    for record_index, (expected_role, processed_record) in enumerate(
        zip(expected_roles, processed_valid)
    ):
        actual_role = processed_record.get("role")
        if actual_role != expected_role:
            role_mismatches.append(
                (record_index, expected_role, actual_role)
            )
            if len(role_mismatches) >= 10:
                break

    if role_mismatches:
        raise RuntimeError(
            "Raw-to-processed role sequence verification failed. "
            f"First mismatches: {role_mismatches}"
        )

    if len(set(record_conv_ids)) != len(raw_conv_ids):
        raise RuntimeError(
            "Not every raw validation conversation contributed at least one "
            "processed record, contrary to the verified dataset diagnostic."
        )

    print("\n========== CONVERSATION ID RECONSTRUCTION ==========")
    print(f"Raw validation conversations: {len(raw_conversations)}")
    print(f"Processed validation records: {len(processed_valid)}")
    print(f"Reconstructed record IDs: {len(record_conv_ids)}")
    print(f"Unique reconstructed conversation IDs: {len(set(record_conv_ids))}")
    print("CRSLab augmentation boundary check: PASS")
    print("Record-by-record role sequence check: PASS")

    return record_conv_ids


def align_events(valid_data, mapping, record_conv_ids):
    """Align all recommender target events to the frozen KGSF event mapping."""
    required = {"event_id", "conv_id", "target_movie_id"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"Mapping must contain {sorted(required)}")

    if len(record_conv_ids) != len(valid_data):
        raise RuntimeError(
            f"Conversation-ID array length {len(record_conv_ids)} "
            f"does not match validation records {len(valid_data)}."
        )

    m = mapping.copy()
    for col in required:
        m[col] = pd.to_numeric(m[col], errors="raise").astype("int64")

    if m.event_id.duplicated().any():
        raise ValueError("Duplicate event_id in frozen mapping.")

    expected = Counter(zip(m.conv_id.tolist(), m.target_movie_id.tolist()))
    lookup = defaultdict(list)
    for r in m.sort_values("event_id").itertuples(index=False):
        lookup[(int(r.conv_id), int(r.target_movie_id))].append(int(r.event_id))

    observed = []
    contexts = {}
    for rec_idx, rec in enumerate(valid_data):
        if rec.get("role") != "Recommender":
            continue

        cid = int(record_conv_ids[rec_idx])
        if "items" not in rec or "context_entities" not in rec:
            raise KeyError(
                f"Recommender record {rec_idx} lacks items/context_entities; "
                f"keys={sorted(rec.keys())}"
            )

        contexts[rec_idx] = rec["context_entities"]
        for item in rec["items"]:
            observed.append((cid, int(item), rec_idx))

    actual = Counter((cid, target) for cid, target, _ in observed)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise RuntimeError(
            "KBRD validation events do not exactly match frozen KGSF mapping. "
            f"Expected={sum(expected.values())}, observed={sum(actual.values())}; "
            f"missing examples={list(missing.items())[:8]}; "
            f"unexpected examples={list(extra.items())[:8]}. "
            "No recommendation export will be written."
        )

    # When a (conversation, target) pair occurs more than once, preserve
    # occurrence order and assign the corresponding event IDs in ascending
    # event_id order. The complete multiset equality above is mandatory.
    used = Counter()
    events = []
    for cid, target, rec_idx in observed:
        key = (cid, target)
        eid = lookup[key][used[key]]
        used[key] += 1
        events.append({
            "event_id": eid,
            "conv_id": cid,
            "target_movie_id": target,
            "source_record_index": rec_idx,
        })

    events.sort(key=lambda x: x["event_id"])
    if len(events) != len(m) or len({e["event_id"] for e in events}) != len(events):
        raise RuntimeError("Event mapping is not one-to-one.")

    mapped_conversations = set(m["conv_id"].astype(int).tolist())
    reconstructed_conversations = set(record_conv_ids)
    absent_conversations = mapped_conversations - reconstructed_conversations
    if absent_conversations:
        raise RuntimeError(
            "Some frozen mapping conversation IDs are absent from raw validation: "
            f"{sorted(absent_conversations)[:20]}"
        )

    print("\n========== FROZEN EVENT ALIGNMENT ==========")
    print(f"Frozen mapping events: {len(m)}")
    print(f"Observed KBRD validation target events: {len(observed)}")
    print(f"Unique mapped conversations: {len(mapped_conversations)}")
    print("Exact (conv_id, target_movie_id) multiset match: PASS")

    return events, contexts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/crs/kbrd/redial.yaml")
    ap.add_argument("--gpu", default="-1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--checkpoint", default="save/KBRD.pth")
    ap.add_argument("--catalog", default="catalog_audit_corrected.pkl")
    ap.add_argument("--mapping", default="results_corrected/m2_event_conversation_mapping.csv")
    ap.add_argument("--raw-valid", default="data/dataset/redial/nltk/valid_data.json")
    ap.add_argument("--out-dir", default="results_corrected/experiment_22_04B")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--expected-raw-conversations", type=int, default=901)
    ap.add_argument("--expected-validation-records", type=int, default=10256)
    args = ap.parse_args()

    if args.batch_size < 1 or args.top_k < 50:
        raise ValueError("batch-size must be positive and top-k must be >= 50")

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "-1" if args.gpu.strip() == "-1" else args.gpu
    )
    seed_all(args.seed)

    from crslab.config import Config
    from crslab.data import get_dataset, get_dataloader
    from crslab.system import get_system

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    catalog_path = Path(args.catalog)
    mapping_path = Path(args.mapping)
    raw_valid_path = Path(args.raw_valid)

    for p, label in (
        (config_path, "config"),
        (checkpoint_path, "KBRD checkpoint"),
        (catalog_path, "catalog"),
        (mapping_path, "event mapping"),
        (raw_valid_path, "raw validation JSON"),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    config = Config(str(config_path), args.gpu)
    if str(config["dataset"]).lower() != "redial" or config["model_name"] != "KBRD":
        raise ValueError("This exporter requires the ReDial KBRD configuration.")

    print("=" * 76)
    print("EXPERIMENT 22.4B — KBRD RECOMMENDATION EXPORT")
    print("=" * 76)

    dataset = get_dataset(config, config["tokenize"], restore=True, save=False)
    vocab, side_data = dataset.vocab, dataset.side_data

    train_loader = get_dataloader(config, dataset.train_data, vocab)
    valid_loader = get_dataloader(config, dataset.valid_data, vocab)
    # CRSLab factory requires this argument; it is never iterated.
    test_loader = get_dataloader(config, dataset.test_data, vocab)
    system = get_system(
        config, train_loader, valid_loader, test_loader, vocab, side_data,
        restore_system=False, interact=False, debug=False, tensorboard=False
    )
    device = system.device

    state = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if "model_state_dict" not in state:
        raise RuntimeError(f"Unexpected checkpoint keys: {list(state.keys())}")
    system.model.load_state_dict(state["model_state_dict"], strict=True)
    system.model.eval()

    with catalog_path.open("rb") as f:
        catalog = pickle.load(f)  # trusted local project artifact

    movie_ids = [int(x) for x in side_data["item_entity_ids"]]
    if movie_ids != [int(x) for x in catalog["movie_ids"]]:
        raise RuntimeError(
            "KBRD movie entity ordering differs from frozen catalog."
        )
    if len(movie_ids) != 6924 or len(set(movie_ids)) != 6924:
        raise RuntimeError(
            f"Expected 6,924 unique catalog movies; got {len(movie_ids)}."
        )
    movie_index = torch.tensor(movie_ids, dtype=torch.long, device=device)

    # Reconstruct the conversation ID for every flattened CRSLab validation
    # record using the original raw conversation boundaries.
    record_conv_ids = reconstruct_validation_conversation_ids(
        raw_valid_path,
        dataset.valid_data,
        expected_raw_conversations=args.expected_raw_conversations,
        expected_processed_records=args.expected_validation_records,
    )

    mapping = pd.read_csv(mapping_path)
    events, contexts = align_events(
        dataset.valid_data, mapping, record_conv_ids
    )
    if len(events) != 3010:
        raise RuntimeError(
            f"Expected 3,010 validation events; got {len(events)}."
        )

    print("\n========== EXPORT CONFIGURATION ==========")
    print(f"Validation records: {len(dataset.valid_data)}")
    print(f"Aligned events: {len(events)}")
    print(f"Unique event conversations: {len({e['conv_id'] for e in events})}")
    print(f"Catalog movies: {len(movie_ids)}")
    print(f"Device: {device}")
    print("Test loader: constructed for CRSLab compatibility only.")
    print("Test loader iterated: NO")

    n_events, n_movies = len(events), len(movie_ids)
    score_matrix = np.empty((n_events, n_movies), dtype=np.float32)
    target_ranks = []
    movie_position = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    for start in tqdm(
        range(0, n_events, args.batch_size), desc="KBRD scoring"
    ):
        stop = min(start + args.batch_size, n_events)
        chunk = events[start:stop]
        batch = {
            "context_entities": [
                contexts[e["source_record_index"]] for e in chunk
            ],
            "item": torch.tensor(
                [e["target_movie_id"] for e in chunk],
                dtype=torch.long,
                device=device,
            ),
        }

        with torch.no_grad():
            _, all_entity_scores = system.model.forward(
                batch, mode="test", stage="rec"
            )

        if (
            all_entity_scores.ndim != 2
            or all_entity_scores.shape[0] != len(chunk)
        ):
            raise RuntimeError(
                f"Unexpected KBRD output shape: {tuple(all_entity_scores.shape)}"
            )
        if max(movie_ids) >= all_entity_scores.shape[1]:
            raise RuntimeError(
                "Movie entity ID exceeds KBRD score dimension."
            )

        scores = (
            all_entity_scores.index_select(1, movie_index)
            .float()
            .cpu()
            .numpy()
        )
        if (
            scores.shape != (len(chunk), n_movies)
            or not np.isfinite(scores).all()
        ):
            raise RuntimeError(
                "Invalid or non-finite KBRD movie score matrix."
            )
        score_matrix[start:stop] = scores

    # Rankings use model scores only. Targets are used only for evaluation.
    top_k = min(args.top_k, n_movies)
    top_rows = []

    for i, event in enumerate(events):
        order = np.lexsort(
            (np.asarray(movie_ids, dtype=np.int64), -score_matrix[i])
        )
        target = int(event["target_movie_id"])
        if target not in movie_position:
            raise RuntimeError(
                f"Target movie {target} is absent from frozen catalog."
            )
        target_pos = movie_position[target]
        rank = int(np.flatnonzero(order == target_pos)[0]) + 1
        target_ranks.append(rank)

        top = order[:top_k]
        top_rows.append({
            "event_id": event["event_id"],
            "conv_id": event["conv_id"],
            "target_movie_id": target,
            "target_rank_full_catalog": rank,
            "top_k": top_k,
            "top_movie_ids": json.dumps(
                [movie_ids[j] for j in top], separators=(",", ":")
            ),
            "top_scores": json.dumps(
                [float(score_matrix[i, j]) for j in top],
                separators=(",", ":"),
            ),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_path = out_dir / "kbrd_validation_full_scores.npz"
    np.savez_compressed(
        score_path,
        event_id=np.asarray(
            [e["event_id"] for e in events], dtype=np.int64
        ),
        conv_id=np.asarray(
            [e["conv_id"] for e in events], dtype=np.int64
        ),
        target_movie_id=np.asarray(
            [e["target_movie_id"] for e in events], dtype=np.int64
        ),
        movie_ids=np.asarray(movie_ids, dtype=np.int64),
        scores=score_matrix,
    )

    top_path = out_dir / f"kbrd_validation_top{top_k}.csv"
    pd.DataFrame(top_rows).to_csv(top_path, index=False)

    manifest_path = out_dir / "kbrd_validation_event_manifest.csv"
    pd.DataFrame([
        {
            **event,
            "score_row": i,
            "target_rank_full_catalog": target_ranks[i],
        }
        for i, event in enumerate(events)
    ]).to_csv(manifest_path, index=False)

    # Reload and verify the saved artifact before declaring success.
    with np.load(score_path, allow_pickle=False) as check:
        if check["scores"].shape != (3010, 6924):
            raise RuntimeError(
                f"Saved matrix has wrong shape: {check['scores'].shape}"
            )
        if not np.array_equal(
            check["event_id"],
            np.asarray([e["event_id"] for e in events], dtype=np.int64),
        ):
            raise RuntimeError("Saved event order failed verification.")
        if not np.array_equal(
            check["conv_id"],
            np.asarray([e["conv_id"] for e in events], dtype=np.int64),
        ):
            raise RuntimeError("Saved conversation IDs failed verification.")
        if not np.array_equal(
            check["movie_ids"], np.asarray(movie_ids, dtype=np.int64)
        ):
            raise RuntimeError("Saved movie order failed verification.")
        if not np.isfinite(check["scores"]).all():
            raise RuntimeError("Saved score matrix contains non-finite values.")

    summary = {
        "experiment": "22.4B_KBRD_recommendation_export",
        "dataset": "ReDial",
        "backbone": "KBRD",
        "checkpoint": str(checkpoint_path),
        "checkpoint_loaded_strict": True,
        "split": "validation_only",
        "raw_validation_conversations": args.expected_raw_conversations,
        "processed_validation_records": len(dataset.valid_data),
        "reconstructed_record_conversation_ids": len(record_conv_ids),
        "record_role_sequence_verified": True,
        "events": n_events,
        "unique_conversations": len({e["conv_id"] for e in events}),
        "catalog_items": n_movies,
        "score_matrix_shape": list(score_matrix.shape),
        "top_k": top_k,
        "alignment": (
            "raw conv_id reconstructed from exact CRSLab augmentation "
            "boundaries; exact multiset match on (conv_id, target_movie_id)"
        ),
        "alignment_passed": True,
        "target_used_for_ranking": False,
        "target_rank_hit10_validation_diagnostic": float(
            np.mean(np.asarray(target_ranks) <= 10)
        ),
        "target_rank_hit50_validation_diagnostic": float(
            np.mean(np.asarray(target_ranks) <= 50)
        ),
        "test_loader_constructed": True,
        "test_loader_iterated": False,
        "test_split_evaluated": False,
        "files": {
            "full_scores": str(score_path),
            "top_candidates": str(top_path),
            "event_manifest": str(manifest_path),
        },
    }

    summary_path = out_dir / "kbrd_validation_export_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("EXPERIMENT 22.4B COMPLETE")
    print("=" * 76)
    print(f"Events exported: {n_events}")
    print(f"Unique conversations: {summary['unique_conversations']}")
    print(f"Catalog: {n_movies} movies")
    print(f"Full score matrix: {score_matrix.shape}")
    print(
        "Validation Hit@10 diagnostic: "
        f"{summary['target_rank_hit10_validation_diagnostic']:.6f}"
    )
    print(
        "Validation Hit@50 diagnostic: "
        f"{summary['target_rank_hit50_validation_diagnostic']:.6f}"
    )
    print(f"Full scores: {score_path}")
    print(f"Top candidates: {top_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print("Conversation reconstruction: PASS")
    print("Event alignment: PASS")
    print("Test split evaluated: NO")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEXPERIMENT 22.4B FAILED: {exc}", file=sys.stderr)
        raise
