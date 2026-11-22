
#!/usr/bin/env python
"""
Experiment 22.35F:
Category Support and Fairness Feasibility Audit.

Diagnostic only:
- No training
- No retrieval
- No reranking
- No parameter selection
- No independent-test access

Uses frozen KGSF candidates and existing R0/R1 relevance.
"""

from collections import defaultdict
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SCRIPT_22_35E = Path(
    "22_35E_integrated_kg_faircrs_reranking.py"
)

OUTPUT = Path(
    "results_corrected/experiment_22_35/"
    "category_support_audit_22_35F"
)

Q_THRESHOLD = 0.005

RMIN_GRID = [0.0, 0.05, 0.10, 0.25, 0.50]

VARIANTS = {
    "R0_filtered": "D_R0",
    "R1_raw": "D_R1",
}


# ---------------------------------------------------------
# LOAD EXISTING EXPERIMENT IMPLEMENTATION
# ---------------------------------------------------------

def load_module(path):

    spec = importlib.util.spec_from_file_location(
        "experiment_22_35E", path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load 22.35E implementation.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------
# CALCULATE CATEGORY-GROUP RELEVANCE MASS
# ---------------------------------------------------------

def calculate_mass(event_ids, candidates, item2group, categories):

    mass = defaultdict(float)

    for eid in sorted(event_ids):

        c = candidates[eid]

        for j, mid in enumerate(c["ids"]):

            q = float(c["q"][j])

            if q < Q_THRESHOLD:
                continue

            mid = int(mid)
            group = int(item2group[mid])

            for category in categories.get(mid, []):

                key = (str(category), group)

                mass[key] += q

    return mass


# ---------------------------------------------------------
# BUILD SUPPORT TABLES
# ---------------------------------------------------------

def build_tables(mass, variant, split):

    category_groups = defaultdict(dict)

    for (category, group), value in mass.items():

        if value > 1e-12:
            category_groups[category][group] = value

    cell_rows = []
    category_rows = []
    summary_rows = []

    for category, group_values in sorted(category_groups.items()):

        total_mass = sum(group_values.values())

        positive_groups = len(group_values)

        category_row = {
            "variant": variant,
            "split": split,
            "category": category,
            "total_relevance_mass": total_mass,
            "positive_popularity_groups": positive_groups,
        }

        for group, value in sorted(group_values.items()):

            cell_rows.append({
                "variant": variant,
                "split": split,
                "category": category,
                "popularity_group": group,
                "R_cg": value,
                "category_total_R": total_mass,
                "group_relevance_share": value / total_mass,
            })

        for rmin in RMIN_GRID:

            supported = [
                g for g, value in group_values.items()
                if value > 1e-12 and value >= rmin
            ]

            active = len(supported) >= 2

            category_row[
                f"groups_supported_Rmin_{rmin:.2f}"
            ] = len(supported)

            category_row[
                f"active_Rmin_{rmin:.2f}"
            ] = active

        category_rows.append(category_row)

    total_categories = len(category_groups)

    for rmin in RMIN_GRID:

        active_categories = 0
        supported_cells = 0
        categories_with_two_groups = 0

        for group_values in category_groups.values():

            supported = [
                value for value in group_values.values()
                if value > 1e-12 and value >= rmin
            ]

            supported_cells += len(supported)

            if len(group_values) >= 2:
                categories_with_two_groups += 1

            if len(supported) >= 2:
                active_categories += 1

        summary_rows.append({
            "variant": variant,
            "split": split,
            "q_threshold": Q_THRESHOLD,
            "R_min": rmin,
            "categories_with_relevance": total_categories,
            "categories_with_two_positive_groups":
                categories_with_two_groups,
            "active_categories": active_categories,
            "supported_category_group_cells": supported_cells,
        })

    return cell_rows, category_rows, summary_rows


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not SCRIPT_22_35E.is_file():
        raise FileNotFoundError(SCRIPT_22_35E)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    exp = load_module(SCRIPT_22_35E)

    ref = exp.load_reference(exp.D_22_16)

    item2group, catalog, categories = ref.load_metadata(
        exp.D_CAT,
        exp.D_PROFILES
    )

    events, c0 = exp.load_data(
        exp.D_KGSF,
        exp.D_R0,
        catalog
    )

    events1, c1 = exp.load_data(
        exp.D_KGSF,
        exp.D_R1,
        catalog
    )

    if events != events1:
        raise RuntimeError("R0/R1 event mappings differ.")

    conversations = [
        value["conv"] for value in events.values()
    ]

    warm, tune, heldout = ref.split_conversations(
        conversations
    )

    splits = {
        "warm": sorted(
            e for e, v in events.items()
            if v["conv"] in warm
        ),
        "tune": sorted(
            e for e, v in events.items()
            if v["conv"] in tune
        ),
        "heldout": sorted(
            e for e, v in events.items()
            if v["conv"] in heldout
        ),
    }

    all_cells = []
    all_categories = []
    all_summaries = []

    variants = {
        "R0_filtered": c0,
        "R1_raw": c1,
    }

    print("=" * 72)
    print("EXPERIMENT 22.35F")
    print("CATEGORY SUPPORT AND FAIRNESS FEASIBILITY AUDIT")
    print("=" * 72)

    print(
        f"Events={len(events)} | "
        f"Conversations={len(set(conversations))}"
    )

    print(
        f"Warm={len(warm)} | "
        f"Tuning={len(tune)} | "
        f"Held-out={len(heldout)}"
    )

    print(f"q_threshold={Q_THRESHOLD}")
    print(f"R_min diagnostic grid={RMIN_GRID}")

    for variant, candidates in variants.items():

        for split, event_ids in splits.items():

            mass = calculate_mass(
                event_ids,
                candidates,
                item2group,
                categories
            )

            cells, category_rows, summaries = build_tables(
                mass,
                variant,
                split
            )

            all_cells.extend(cells)
            all_categories.extend(category_rows)
            all_summaries.extend(summaries)

    pd.DataFrame(all_cells).to_csv(
        OUTPUT / "category_group_relevance_mass.csv",
        index=False
    )

    pd.DataFrame(all_categories).to_csv(
        OUTPUT / "category_support_by_split.csv",
        index=False
    )

    summary_df = pd.DataFrame(all_summaries)

    summary_df.to_csv(
        OUTPUT / "category_support_summary.csv",
        index=False
    )

    manifest = {
        "experiment":
            "22.35F_category_support_feasibility_audit",

        "purpose":
            "Diagnose category-popularity-group relevance support",

        "source_experiment":
            "22.35E_integrated_kg_faircrs_reranking",

        "events": len(events),
        "conversations": len(set(conversations)),

        "warm_conversations": len(warm),
        "tuning_conversations": len(tune),
        "heldout_conversations": len(heldout),

        "variants": list(variants.keys()),

        "q_threshold": Q_THRESHOLD,
        "R_min_diagnostic_grid": RMIN_GRID,

        "independent_test_read": False,
        "training_performed": False,
        "retrieval_performed": False,
        "reranking_performed": False,
        "parameter_selection_performed": False,

        "outputs": [
            "category_support_summary.csv",
            "category_support_by_split.csv",
            "category_group_relevance_mass.csv",
        ],
    }

    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\n========== SUPPORT SUMMARY ==========")

    print(
        summary_df[
            [
                "variant",
                "split",
                "R_min",
                "categories_with_relevance",
                "categories_with_two_positive_groups",
                "active_categories",
                "supported_category_group_cells",
            ]
        ].to_string(index=False)
    )

    print("\n========== 22.35F COMPLETE ==========")
    print("Output:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
