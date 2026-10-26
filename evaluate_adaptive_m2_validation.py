
"""Phase 5.5: validation-only adaptive KGSF + KG rank fusion.

A0: fixed alpha=0.25.
A1: candidate-frequency-adaptive KG weight.
A2: frequency-adaptive weight multiplied by context KG coverage.

Ground-truth labels are used only for evaluation, never for ranking.
No model training, checkpoint changes, or test evaluation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog
from evaluate_kg_retrieval import load_graphs, build_index, retrieve


def gini(values):
    a = np.sort(np.asarray(values, dtype=np.float64))
    if len(a) == 0 or a.sum() == 0:
        return 0.0
    n = len(a)
    return float(
        2.0 * np.dot(np.arange(1, n + 1), a)
        / (n * a.sum()) - (n + 1.0) / n
    )


def make_stats(n_movies):
    return {
        "events": 0,
        "hit10": 0,
        "hit50": 0,
        "ndcg10_sum": 0.0,
        "zero_n": 0,
        "zero_hit10": 0,
        "one_n": 0,
        "one_hit10": 0,
        "overlap_n": 0,
        "overlap_hit10": 0,
        "nonoverlap_n": 0,
        "nonoverlap_hit10": 0,
        "exposures": np.zeros(n_movies, dtype=np.int64),
        "group_counts": np.zeros(5, dtype=np.int64),
    }


def update_stats(stats, order, true_local, frequency,
                 target_overlap, group_ids):
    """Update metrics using the ranked movie indices."""
    stats["events"] += 1

    positions = np.flatnonzero(order == true_local)
    rank = int(positions[0]) + 1 if len(positions) else None

    if rank is not None:
        if rank <= 50:
            stats["hit50"] += 1
        if rank <= 10:
            stats["hit10"] += 1
            stats["ndcg10_sum"] += 1.0 / np.log2(rank + 1)

    if frequency == 0:
        stats["zero_n"] += 1
        stats["zero_hit10"] += int(rank is not None and rank <= 10)

    if frequency == 1:
        stats["one_n"] += 1
        stats["one_hit10"] += int(rank is not None and rank <= 10)

    segment = "overlap" if target_overlap else "nonoverlap"
    stats[f"{segment}_n"] += 1
    stats[f"{segment}_hit10"] += int(rank is not None and rank <= 10)

    top10 = order[:10]
    np.add.at(stats["exposures"], top10, 1)
    np.add.at(stats["group_counts"], group_ids[top10], 1)


def summarize(name, stats, n_events, n_movies):
    slots = int(stats["exposures"].sum())
    if slots != n_events * 10:
        raise RuntimeError(f"{name}: top-10 exposure accounting mismatch")
    if int(stats["group_counts"].sum()) != slots:
        raise RuntimeError(f"{name}: group exposure accounting mismatch")

    def rate(hits, count):
        return hits / count if count else None

    return {
        "method": name,
        "events": stats["events"],
        "hit10": rate(stats["hit10"], n_events),
        "hit50": rate(stats["hit50"], n_events),
        "ndcg10": rate(stats["ndcg10_sum"], n_events),
        "zero_shot_events": stats["zero_n"],
        "zero_shot_hit10": rate(
            stats["zero_hit10"], stats["zero_n"]
        ),
        "one_shot_events": stats["one_n"],
        "one_shot_hit10": rate(
            stats["one_hit10"], stats["one_n"]
        ),
        "target_context_overlap_events": stats["overlap_n"],
        "overlap_hit10": rate(
            stats["overlap_hit10"], stats["overlap_n"]
        ),
        "nonoverlap_events": stats["nonoverlap_n"],
        "nonoverlap_hit10": rate(
            stats["nonoverlap_hit10"], stats["nonoverlap_n"]
        ),
        "item_gini": gini(stats["exposures"]),
        "catalog_coverage10": float(
            np.count_nonzero(stats["exposures"]) / n_movies
        ),
        "group_exposure_shares": (
            stats["group_counts"] / slots
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/crs/kgsf/redial.yaml")
    parser.add_argument("--catalog", default="catalog_audit_corrected.pkl")
    parser.add_argument("--raw-dir", default="data/dataset/redial/nltk")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hub-cutoff", type=int, default=500)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--base-alpha", type=float, default=0.25)
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=[1.0, 3.0, 7.0],
        help="Frequency-adaptation strengths selected on validation."
    )
    parser.add_argument(
        "--out",
        default="results_corrected/adaptive_m2_validation.json"
    )
    args = parser.parse_args()

    if args.base_alpha < 0 or args.rrf_k <= 0 or args.hub_cutoff <= 0:
        raise ValueError("Alpha must be nonnegative; cutoffs must be positive.")
    if not args.strengths or any(x < 0 for x in args.strengths):
        raise ValueError("Adaptation strengths must be nonnegative.")

    strengths = list(dict.fromkeys(args.strengths))

    system, valid, _test, side_data = initialize(
        args.config, args.gpu
    )
    catalog = load_catalog(
        args.catalog, side_data["item_entity_ids"], system.device
    )

    movie_ids = np.asarray(catalog["movie_ids"], dtype=np.int64)
    n_movies = len(movie_ids)
    pos = {int(movie): i for i, movie in enumerate(movie_ids)}

    freq_dict = {
        int(k): int(v)
        for k, v in catalog["meta"]["freq_dict"].items()
    }
    movie_freqs = np.asarray(
        [freq_dict[int(movie)] for movie in movie_ids],
        dtype=np.float64
    )

    raw_adj = load_graphs(
        side_data, Path(args.raw_dir)
    )["R1_raw"]
    index = build_index(
        raw_adj, movie_ids.tolist(), args.hub_cutoff
    )

    pad = int(valid.pad_entity_idx)
    group_ids = np.asarray(catalog["group_ids"], dtype=np.int64)

    # Candidate-specific frequency multiplier. All variants share
    # identical KGSF scores, KG retrieval, and validation events.
    freq_factors = {
        strength: (
            1.0 + strength / np.sqrt(1.0 + movie_freqs)
        )
        for strength in strengths
    }

    variant_names = ["A0_fixed_025"]
    for strength in strengths:
        tag = f"{strength:g}"
        variant_names.extend([
            f"A1_frequency_{tag}",
            f"A2_frequency_context_{tag}",
        ])

    stats = {
        name: make_stats(n_movies) for name in variant_names
    }

    n_events = 0
    system.model.eval()

    with torch.no_grad():
        batches = valid.get_rec_data(
            batch_size=args.batch_size, shuffle=False
        )

        for batch in tqdm(batches, desc="Adaptive M2 validation"):
            model_batch = tuple(t.to(system.device) for t in batch)
            output = system.model.recommend(model_batch, mode="test")

            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Unexpected KGSF recommendation output.")

            scores = output[2]
            scores = scores.index_select(
                1, catalog["candidate_ids"]
            ).detach().cpu().numpy()

            if scores.ndim != 2 or scores.shape[1] != n_movies:
                raise RuntimeError("Movie score shape mismatch.")

            for context_row, target, row_scores in zip(
                batch[0].tolist(), batch[-1].tolist(), scores
            ):
                target = int(target)
                if target not in pos:
                    raise RuntimeError(
                        f"Target {target} missing from candidate catalog."
                    )

                context = [
                    int(entity) for entity in context_row
                    if int(entity) != pad
                ]

                # Context confidence is the fraction of unique context
                # entities that have at least one raw KG neighbor.
                unique_context = set(context)
                context_confidence = (
                    sum(bool(raw_adj.get(entity, ()))
                        for entity in unique_context)
                    / len(unique_context)
                    if unique_context else 0.0
                )

                kg_ranked = retrieve(context, index)

                # Rank all eligible movies using KGSF scores.
                kgsf_order = np.lexsort((movie_ids, -row_scores))
                kgsf_rank = np.empty(n_movies, dtype=np.int32)
                kgsf_rank[kgsf_order] = np.arange(
                    1, n_movies + 1, dtype=np.int32
                )
                kgsf_vote = 1.0 / (
                    args.rrf_k + kgsf_rank.astype(np.float64)
                )

                kg_vote = np.zeros(n_movies, dtype=np.float64)
                for rank, movie in enumerate(kg_ranked, start=1):
                    local = pos.get(int(movie))
                    if local is not None:
                        kg_vote[local] = 1.0 / (args.rrf_k + rank)

                true_local = pos[target]
                frequency = freq_dict[target]
                target_overlap = target in unique_context

                # A0: fixed-weight reference.
                fused = kgsf_vote + args.base_alpha * kg_vote
                order = np.lexsort((movie_ids, -fused))[:50]
                update_stats(
                    stats["A0_fixed_025"], order, true_local,
                    frequency, target_overlap, group_ids
                )

                # A1/A2: candidate-specific weights.
                for strength in strengths:
                    factor = freq_factors[strength]
                    tag = f"{strength:g}"

                    # A1: frequency-adaptive.
                    alpha_a1 = args.base_alpha * factor
                    fused_a1 = kgsf_vote + alpha_a1 * kg_vote
                    order_a1 = np.lexsort(
                        (movie_ids, -fused_a1)
                    )[:50]
                    update_stats(
                        stats[f"A1_frequency_{tag}"],
                        order_a1, true_local, frequency,
                        target_overlap, group_ids
                    )

                    # A2: frequency and context-confidence adaptive.
                    alpha_a2 = (
                        args.base_alpha * factor * context_confidence
                    )
                    fused_a2 = kgsf_vote + alpha_a2 * kg_vote
                    order_a2 = np.lexsort(
                        (movie_ids, -fused_a2)
                    )[:50]
                    update_stats(
                        stats[f"A2_frequency_context_{tag}"],
                        order_a2, true_local, frequency,
                        target_overlap, group_ids
                    )

                n_events += 1

    if n_events != 3010:
        raise RuntimeError(
            f"Expected 3010 validation events, observed {n_events}."
        )

    results = [
        summarize(name, stats[name], n_events, n_movies)
        for name in variant_names
    ]

    report = {
        "split": "validation_only",
        "method": "Fixed versus candidate-frequency and context-adaptive RRF",
        "events": n_events,
        "candidate_movies": n_movies,
        "base_alpha": args.base_alpha,
        "rrf_k": args.rrf_k,
        "hub_cutoff": args.hub_cutoff,
        "frequency_strengths": strengths,
        "context_confidence": (
            "Fraction of unique context entities with at least one raw KG neighbor."
        ),
        "target_used_for_ranking": False,
        "checkpoint_changed": False,
        "test_split_evaluated": False,
        "results": results,
        "caution": (
            "Exploratory validation comparison, not a novelty claim. "
            "Select configurations on validation only, then evaluate the "
            "locked method on test once. Report uncertainty and ablations."
        ),
    }

    target_path = Path(args.out)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Saved:", target_path.resolve())


if __name__ == "__main__":
    main()
