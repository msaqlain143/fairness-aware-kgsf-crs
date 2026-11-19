
#!/usr/bin/env python
"""
Experiment 22.35D
Relevance Distribution and Eligibility Feasibility Audit

Validation-only.
- Uses frozen 22.35C relevance outputs.
- Uses the exact 22.16 category evidence and conversation split functions.
- Does not train, retrieve, or modify KGSF.
- Does not use targets to construct relevance or eligibility.
- Does not evaluate the independent test set.
- Does not select a final threshold automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
INPUT_DIR = (
    ROOT / "results_corrected/experiment_22_35"
)

RELEVANCE_DIR = INPUT_DIR / "relevance_validation"
MANIFEST_DIR = INPUT_DIR / "kgsf_validation"

DEFAULT_R0 = (
    RELEVANCE_DIR / "event_candidate_q_ei_R0_filtered.csv"
)

DEFAULT_R1 = (
    RELEVANCE_DIR / "event_candidate_q_ei_R1_raw.csv"
)

DEFAULT_22_16 = ROOT / "22_16_category_context_utility_calibration.py"

DEFAULT_CATALOG = ROOT / "catalog_audit_corrected.pkl"

DEFAULT_PROFILES = (
    ROOT
    / "results_category_audit/experiment_21_22A/"
    / "catalog_item_category_profiles.csv"
)

DEFAULT_OUT = INPUT_DIR / "eligibility_feasibility"

EXPECTED_EVENTS = 3010
CANDIDATES_PER_EVENT = 500

EPS = 1e-12

# Diagnostic grid only.
# These are NOT final selected eligibility thresholds.
THRESHOLDS = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050,
)


def load_22_16(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "experiment_22_16",
        path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "load_metadata",
        "split_conversations",
    ]

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"22.16 missing functions: {missing}"
        )

    return module


def find_manifest(directory):

    candidates = list(
        directory.glob("*event_manifest*.csv")
    )

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one validation event manifest. "
            f"Found: {candidates}"
        )

    return candidates[0]


def validate_relevance(df, label):

    required = {
        "event_id",
        "candidate_rank",
        "movie_id",
        "k_gsf_score",
        "k_gsf_candidate_probability",
        "kg_rank",
        "kg_confidence",
        "q_ei",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {sorted(missing)}"
        )

    if df.duplicated(
        ["event_id", "movie_id"]
    ).any():
        raise RuntimeError(
            f"{label}: duplicate event-movie pairs."
        )

    if len(df) != EXPECTED_EVENTS * CANDIDATES_PER_EVENT:
        raise RuntimeError(
            f"{label}: unexpected row count {len(df)}"
        )

    counts = df.groupby("event_id").size()

    if len(counts) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"{label}: invalid event count."
        )

    if not counts.eq(CANDIDATES_PER_EVENT).all():
        raise RuntimeError(
            f"{label}: invalid candidates per event."
        )

    numeric = [
        "k_gsf_score",
        "k_gsf_candidate_probability",
        "kg_rank",
        "kg_confidence",
        "q_ei",
    ]

    if not np.isfinite(
        df[numeric].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"{label}: non-finite numeric values."
        )

    if (df["q_ei"] < 0).any():
        raise RuntimeError(
            f"{label}: negative q values."
        )

    if (df["kg_rank"] < 0).any():
        raise RuntimeError(
            f"{label}: negative KG ranks."
        )

    mass = df.groupby("event_id")["q_ei"].sum()

    if not np.allclose(
        mass.to_numpy(),
        1.0,
        atol=1e-8
    ):
        raise RuntimeError(
            f"{label}: event relevance normalization failed."
        )

    for event_id, group in df.groupby("event_id"):

        ranks = np.sort(
            group["candidate_rank"].to_numpy()
        )

        if not np.array_equal(
            ranks,
            np.arange(1, CANDIDATES_PER_EVENT + 1)
        ):
            raise RuntimeError(
                f"{label}: invalid candidate ranks at {event_id}"
            )

    return mass


def event_diagnostics(df, events):

    records = []

    for event_id, group in df.groupby(
        "event_id",
        sort=True
    ):

        q = group["q_ei"].to_numpy(
            dtype=np.float64
        )

        kg_supported = (
            group["kg_rank"].to_numpy() > 0
        )

        q_sorted = np.sort(q)[::-1]

        entropy = -float(
            np.sum(q * np.log(q + EPS))
        )

        effective_count = 1.0 / (
            float(np.sum(q ** 2)) + EPS
        )

        event_info = events.loc[int(event_id)]

        records.append({

            "event_id": int(event_id),

            "conv_id": int(event_info.conv_id),
            "split": str(event_info.split),
            "candidate_count": len(q),

            "q_max": float(q.max()),

            "q_median": float(np.median(q)),

            "q_min": float(q.min()),

            "entropy": entropy,

            "normalized_entropy": (
                entropy / math.log(len(q))
            ),

            "effective_candidate_count": (
                effective_count
            ),

            "top1_mass": float(q_sorted[:1].sum()),

            "top5_mass": float(q_sorted[:5].sum()),

            "top10_mass": float(q_sorted[:10].sum()),

            "kg_supported_count": int(
                kg_supported.sum()
            ),

            "kg_unsupported_count": int(
                (~kg_supported).sum()
            ),

            "kg_supported_relevance_mass": float(
                q[kg_supported].sum()
            ),

            "kg_unsupported_relevance_mass": float(
                q[~kg_supported].sum()
            ),

        })

    return pd.DataFrame(records)


def distribution_summary(event_df, variant):

    metrics = [
        "q_max",
        "q_median",
        "entropy",
        "normalized_entropy",
        "effective_candidate_count",
        "top1_mass",
        "top5_mass",
        "top10_mass",
        "kg_supported_count",
        "kg_supported_relevance_mass",
        "kg_unsupported_relevance_mass",
    ]

    rows = []

    for split in [
        "warm",
        "tune",
        "heldout",
        "all",
    ]:

        if split == "all":
            part = event_df
        else:
            part = event_df[
                event_df["split"] == split
            ]

        if part.empty:
            continue

        for metric in metrics:

            values = part[metric]

            rows.append({

                "variant": variant,

                "split": split,

                "metric": metric,

                "mean": float(values.mean()),

                "median": float(values.median()),

                "std": float(values.std()),

                "p10": float(values.quantile(0.10)),

                "p90": float(values.quantile(0.90)),

            })

    return pd.DataFrame(rows)


def eligibility_audit(
    df,
    events,
    categories,
    item2group,
    variant
):

    records = []
    rcg_records = []

    joined = df.merge(
        events[["conv_id", "split"]],
        left_on="event_id",
        right_index=True,
        validate="many_to_one"
    )

    for threshold in THRESHOLDS:

        for kg_gate in [
            False,
            True,
        ]:

            gate_name = (
                "q_threshold_and_kg_support"
                if kg_gate
                else "q_threshold_only"
            )

            for (event_id, split), group in joined.groupby(
                ["event_id", "split"],
                sort=True
            ):

                q = group["q_ei"].to_numpy(
                    dtype=np.float64
                )

                ranks = group["kg_rank"].to_numpy(
                    dtype=np.int64
                )

                ids = group["movie_id"].to_numpy(
                    dtype=np.int64
                )

                eligible = q >= threshold

                if kg_gate:
                    eligible &= ranks > 0

                selected = np.flatnonzero(
                    eligible
                )

                categorized = 0
                eligible_categories = set()

                for index in selected:

                    movie_id = int(ids[index])

                    movie_categories = categories.get(
                        movie_id,
                        []
                    )

                    if not movie_categories:
                        continue

                    categorized += 1

                    group_id = int(
                        item2group[movie_id]
                    )

                    contribution = (
                        float(q[index])
                        / len(movie_categories)
                    )

                    for category in movie_categories:

                        eligible_categories.add(category)

                        rcg_records.append({

                            "variant": variant,

                            "split": split,

                            "threshold": threshold,

                            "eligibility_rule": gate_name,

                            "category": category,

                            "popularity_group": group_id,

                            "R_cg": contribution,

                        })

                selected_q_mass = float(
                    q[selected].sum()
                )

                records.append({

                    "variant": variant,

                    "split": split,

                    "event_id": int(event_id),

                    "threshold": threshold,

                    "eligibility_rule": gate_name,

                    "eligible_candidates": len(selected),

                    "eligible_fraction": (
                        len(selected) / len(q)
                    ),

                    "eligible_relevance_mass": (
                        selected_q_mass
                    ),

                    "categorized_eligible_candidates": (
                        categorized
                    ),

                    "eligible_category_count": (
                        len(eligible_categories)
                    ),

                })

    return (
        pd.DataFrame(records),
        pd.DataFrame(rcg_records)
    )


def summarize_eligibility(event_results):

    keys = [
        "variant",
        "split",
        "threshold",
        "eligibility_rule",
    ]

    return (
        event_results
        .groupby(keys, as_index=False)
        .agg(
            events=("event_id", "nunique"),

            mean_eligible_candidates=(
                "eligible_candidates", "mean"
            ),

            median_eligible_candidates=(
                "eligible_candidates", "median"
            ),

            p10_eligible_candidates=(
                "eligible_candidates",
                lambda x: x.quantile(0.10)
            ),

            p90_eligible_candidates=(
                "eligible_candidates",
                lambda x: x.quantile(0.90)
            ),

            mean_eligible_fraction=(
                "eligible_fraction", "mean"
            ),

            mean_eligible_relevance_mass=(
                "eligible_relevance_mass", "mean"
            ),

            mean_categorized_eligible_candidates=(
                "categorized_eligible_candidates",
                "mean"
            ),

            mean_eligible_category_count=(
                "eligible_category_count", "mean"
            ),

            zero_eligible_events=(
                "eligible_candidates",
                lambda x: int((x == 0).sum())
            ),

        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--r0",
        default=str(DEFAULT_R0)
    )

    parser.add_argument(
        "--r1",
        default=str(DEFAULT_R1)
    )

    parser.add_argument(
        "--manifest",
        default=None
    )

    parser.add_argument(
        "--reference",
        default=str(DEFAULT_22_16)
    )

    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG)
    )

    parser.add_argument(
        "--profiles",
        default=str(DEFAULT_PROFILES)
    )

    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT)
    )

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EXPERIMENT 22.35D")
    print("RELEVANCE DISTRIBUTION AND ELIGIBILITY FEASIBILITY")
    print("=" * 76)

    # ---------------------------------------------------------
    # 1. Load original 22.16 infrastructure
    # ---------------------------------------------------------

    reference = load_22_16(
        Path(args.reference)
    )

    item2group, catalog, categories = (
        reference.load_metadata(
            args.catalog,
            args.profiles
        )
    )

    print("Catalog items:", len(catalog))
    print("Items with accepted categories:", sum(
        bool(categories.get(movie_id, []))
        for movie_id in catalog
    ))

    # ---------------------------------------------------------
    # 2. Load event manifest
    # ---------------------------------------------------------

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else find_manifest(MANIFEST_DIR)
    )

    manifest = pd.read_csv(manifest_path)

    required = {
        "event_id",
        "conv_id",
    }

    if not required.issubset(manifest.columns):
        raise RuntimeError(
            f"Invalid manifest columns: {manifest.columns.tolist()}"
        )

    if manifest["event_id"].duplicated().any():
        raise RuntimeError("Duplicate manifest event IDs.")

    if len(manifest) != EXPECTED_EVENTS:
        raise RuntimeError("Unexpected manifest event count.")

    manifest["event_id"] = manifest.event_id.astype(int)
    manifest["conv_id"] = manifest.conv_id.astype(int)

    manifest = manifest.sort_values(
        "event_id"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Reproduce the existing 22.16 conversation split
    # ---------------------------------------------------------

    warm, tune, heldout = (
        reference.split_conversations(
            manifest.conv_id.tolist()
        )
    )

    if (
        warm & tune
        or warm & heldout
        or tune & heldout
    ):
        raise RuntimeError("Conversation split leakage.")

    split_lookup = {}

    for cid in warm:
        split_lookup[cid] = "warm"

    for cid in tune:
        split_lookup[cid] = "tune"

    for cid in heldout:
        split_lookup[cid] = "heldout"

    manifest["split"] = manifest.conv_id.map(
        split_lookup
    )

    if manifest["split"].isna().any():
        raise RuntimeError("Missing conversation split.")

    events = (
        manifest
        .set_index("event_id")
        [["conv_id", "split"]]
    )

    print("\nCONVERSATION SPLITS")
    print("Warm:", len(warm))
    print("Tune:", len(tune))
    print("Heldout:", len(heldout))

    manifest.to_csv(
        out / "conversation_split_assignments.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 4. Load R0 and R1 relevance
    # ---------------------------------------------------------

    variants = {
        "R0_filtered": Path(args.r0),
        "R1_raw": Path(args.r1),
    }

    frames = {}

    for name, path in variants.items():

        if not path.is_file():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        validate_relevance(df, name)

        if set(df.event_id.unique()) != set(events.index):
            raise RuntimeError(
                f"{name}: event IDs do not match manifest."
            )

        if not df.movie_id.isin(catalog).all():
            raise RuntimeError(
                f"{name}: candidates outside catalog."
            )

        frames[name] = df

        print(
            f"\n{name}: PASS | "
            f"rows={len(df):,} | "
            f"events={df.event_id.nunique()}"
        )

    # ---------------------------------------------------------
    # 5. Exact R0/R1 candidate alignment
    # ---------------------------------------------------------

    r0 = frames["R0_filtered"].sort_values(
        ["event_id", "candidate_rank"]
    ).reset_index(drop=True)

    r1 = frames["R1_raw"].sort_values(
        ["event_id", "candidate_rank"]
    ).reset_index(drop=True)

    alignment = (
        (r0.event_id == r1.event_id)
        & (r0.candidate_rank == r1.candidate_rank)
        & (r0.movie_id == r1.movie_id)
    )

    if not alignment.all():
        raise RuntimeError(
            "R0/R1 candidate ordering or ID alignment failed."
        )

    print("\nR0/R1 frozen candidate alignment: PASS")

    # ---------------------------------------------------------
    # 6. Distribution and eligibility audits
    # ---------------------------------------------------------

    all_event_diagnostics = []
    all_distribution = []
    all_eligibility = []
    all_rcg = []

    for name, df in frames.items():

        event_df = event_diagnostics(
            df,
            events
        )

        event_df["variant"] = name

        all_event_diagnostics.append(
            event_df
        )

        all_distribution.append(
            distribution_summary(
                event_df,
                name
            )
        )

        eligibility, rcg = eligibility_audit(
            df,
            events,
            categories,
            item2group,
            name
        )

        all_eligibility.append(eligibility)
        all_rcg.append(rcg)

    event_diagnostics_df = pd.concat(
        all_event_diagnostics,
        ignore_index=True
    )

    distribution_df = pd.concat(
        all_distribution,
        ignore_index=True
    )

    eligibility_events = pd.concat(
        all_eligibility,
        ignore_index=True
    )

    rcg_df = pd.concat(
        all_rcg,
        ignore_index=True
    )

    eligibility_summary = summarize_eligibility(
        eligibility_events
    )

    # ---------------------------------------------------------
    # 7. Save outputs
    # ---------------------------------------------------------

    event_diagnostics_df.to_csv(
        out / "event_relevance_distribution.csv",
        index=False
    )

    distribution_df.to_csv(
        out / "relevance_distribution_summary.csv",
        index=False
    )

    eligibility_events.to_csv(
        out / "event_eligibility_diagnostics.csv",
        index=False
    )

    eligibility_summary.to_csv(
        out / "eligibility_threshold_summary.csv",
        index=False
    )

    rcg_df.to_csv(
        out / "eligibility_category_group_relevance.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. Final integrity checks
    # ---------------------------------------------------------

    checks = {

        "experiment": "22.35D",

        "split": "validation",

        "events": EXPECTED_EVENTS,

        "unique_conversations": int(
            manifest.conv_id.nunique()
        ),

        "candidate_instances_per_variant": (
            EXPECTED_EVENTS * CANDIDATES_PER_EVENT
        ),

        "R0_R1_candidate_alignment": bool(
            alignment.all()
        ),

        "event_relevance_normalization": True,

        "all_q_values_finite": True,

        "conversation_disjoint_splits": True,

        "category_metadata_reused_from_22_16": True,

        "thresholds_selected": False,

        "target_used_for_eligibility": False,

        "training_performed": False,

        "retrieval_performed": False,

        "independent_test_evaluated": False,

        "ranking_changed": False,

        "outputs": [
            "conversation_split_assignments.csv",
            "event_relevance_distribution.csv",
            "relevance_distribution_summary.csv",
            "event_eligibility_diagnostics.csv",
            "eligibility_threshold_summary.csv",
            "eligibility_category_group_relevance.csv",
        ],

    }

    (out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8"
    )

    manifest_out = {

        **checks,

        "threshold_grid": list(THRESHOLDS),

        "threshold_status": (
            "DIAGNOSTIC_GRID_ONLY_NOT_SELECTED"
        ),

        "split_seed": 2200,

        "split_proportions": {
            "warm": 0.60,
            "tune": 0.20,
            "heldout": 0.20,
        },

        "eligibility_definitions": {

            "q_threshold_only": (
                "q_ei >= threshold"
            ),

            "q_threshold_and_kg_support": (
                "q_ei >= threshold AND kg_rank > 0"
            ),

        },

        "relevance_definition": (
            "Frozen KGSF softmax probability multiplied by "
            "(1 + KG confidence), normalized per event."
        ),

        "category_group_relevance": (
            "Sum of eligible q_ei divided equally across "
            "each item's accepted multi-label categories, "
            "stratified by frozen popularity group."
        ),

    }

    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest_out, indent=2),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # 9. Console summary
    # ---------------------------------------------------------

    print("\n" + "=" * 76)
    print("22.35D COMPLETE")
    print("=" * 76)

    print("\nRELEVANCE DISTRIBUTION")

    print(
        distribution_df[
            (distribution_df.split == "tune")
            & (
                distribution_df.metric.isin([
                    "q_max",
                    "effective_candidate_count",
                    "top1_mass",
                    "top5_mass",
                    "top10_mass",
                    "kg_supported_relevance_mass",
                ])
            )
        ].to_string(index=False)
    )

    print("\nELIGIBILITY FEASIBILITY — TUNING CONVERSATIONS")

    print(
        eligibility_summary[
            eligibility_summary.split == "tune"
        ].to_string(index=False)
    )

    print("\nTraining: NO")
    print("Retrieval: NO")
    print("Target used for q or eligibility: NO")
    print("Independent test evaluated: NO")
    print("Threshold automatically selected: NO")

    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()
