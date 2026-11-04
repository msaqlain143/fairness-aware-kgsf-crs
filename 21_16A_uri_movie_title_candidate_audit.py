
"""
EXPERIMENT 21.16A
URI Movie Title Candidate and Metadata Source Audit

Purpose:
1. Load the canonical 6,924 movie identity table.
2. Reconcile the 812 verified numeric identifiers.
3. Generate conservative title candidates for 6,112 DBpedia URI entities.
4. Identify unique, ambiguous and unmatched title candidates.
5. Preserve metadata provenance.
6. Export reproducible audit results.

Important:
URI-derived title candidates are NOT automatically verified.
No source dataset is modified.
No model training or inference is performed.
"""

import csv
import json
import re
import unicodedata

from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(".")

IDENTITY_FILE = (
    ROOT / "results_category_audit"
    / "experiment_21_15"
    / "unified_movie_entity_mapping.csv"
)

MENTION_FILE = (
    ROOT / "external_validation"
    / "redial_reference"
    / "movies_with_mentions.csv"
)

NUMERIC_ALIGNMENT_FILE = (
    ROOT / "results_category_audit"
    / "experiment_21_14"
    / "numeric_movie_semantic_alignment.csv"
)

OUTPUT = (
    ROOT / "results_category_audit"
    / "experiment_21_16A"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. UTILITIES
# ============================================================

def require(condition, message):

    if not condition:
        raise RuntimeError(message)


def read_csv(path):

    require(path.exists(), f"Missing input file: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(csv.DictReader(f))


def write_csv(path, rows, fields):

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)


def clean_uri(uri):

    uri = str(uri or "").strip()

    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]

    return unquote(uri)


def uri_local_label(uri):

    uri = clean_uri(uri)

    if "/resource/" not in uri:
        return ""

    label = uri.split("/resource/", 1)[1]

    # Preserve years while removing DBpedia's
    # explicit film-type suffix.

    label = re.sub(
        r"_\((\d{4})_film\)$",
        r"_(\1)",
        label,
        flags=re.IGNORECASE
    )

    label = re.sub(
        r"_\(film\)$",
        "",
        label,
        flags=re.IGNORECASE
    )

    return label.replace("_", " ")


def normalize_title(value):

    value = unicodedata.normalize(
        "NFKC",
        str(value or "")
    )

    value = value.casefold().strip()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def is_numeric_identity(identity_type):

    return "numeric" in str(
        identity_type
    ).casefold()


# ============================================================
# 3. LOAD SOURCES
# ============================================================

print("=" * 75)
print("EXPERIMENT 21.16A: URI MOVIE TITLE CANDIDATE AUDIT")
print("=" * 75)

identities = read_csv(IDENTITY_FILE)

mentions = read_csv(MENTION_FILE)

numeric_alignment = read_csv(
    NUMERIC_ALIGNMENT_FILE
)

print("\n[1] SOURCE INVENTORY")

print("Canonical identities:", len(identities))
print("Movie reference records:", len(mentions))
print("Numeric alignment records:", len(numeric_alignment))

require(
    len(identities) == 6924,
    "Canonical identity population must equal 6924."
)

require(
    len(mentions) == 6924,
    "Movie reference population must equal 6924."
)

# Check required identity columns.

identity_fields = {
    "movie_entity_id",
    "identity_type",
    "canonical_identifier"
}

require(
    identity_fields.issubset(identities[0].keys()),
    "Unexpected canonical identity schema."
)

mention_fields = {
    "movieId",
    "movieName",
    "nbMentions"
}

require(
    mention_fields.issubset(mentions[0].keys()),
    "Unexpected movie reference schema."
)


# ============================================================
# 4. VALIDATE SOURCE IDENTIFIERS
# ============================================================

print("\n[2] SOURCE INTEGRITY")

internal_ids = [
    row["movie_entity_id"]
    for row in identities
]

reference_ids = [
    str(row["movieId"]).strip()
    for row in mentions
]

require(
    len(set(internal_ids)) == 6924,
    "Duplicate canonical internal IDs."
)

require(
    len(set(reference_ids)) == 6924,
    "Duplicate reference movie IDs."
)

# Build normalized title index.

title_index = defaultdict(list)

reference_by_id = {}

for row in mentions:

    movie_id = str(row["movieId"]).strip()

    movie_title = str(row["movieName"]).strip()

    reference_by_id[movie_id] = row

    title_key = normalize_title(movie_title)

    if title_key:

        title_index[title_key].append(row)

print("Unique internal IDs: PASS")
print("Unique reference IDs: PASS")
print("Reference title index:", len(title_index))


# ============================================================
# 5. LOAD NUMERIC VERIFICATION
# ============================================================

numeric_verified = {}

for row in numeric_alignment:

    external_id = str(
        row["external_numeric_id"]
    ).strip()

    title_match = str(
        row["title_match"]
    ).strip().casefold()

    status = str(
        row["mapping_status"]
    ).strip().casefold()

    numeric_verified[external_id] = (
        title_match == "true"
        and status in {
            "verified",
            "all_verified"
        }
    )

require(
    len(numeric_verified) == 812,
    "Expected 812 unique numeric alignment records."
)

require(
    all(numeric_verified.values()),
    "Numeric semantic verification is incomplete."
)

print("Verified numeric crosswalk:", len(numeric_verified))


# ============================================================
# 6. RECONSTRUCT METADATA CANDIDATES
# ============================================================

print("\n[3] PROCESSING MOVIE IDENTITIES")

complete_inventory = []
uri_candidates = []

counts = {
    "numeric": 0,
    "uri": 0,
    "numeric_verified": 0,
    "uri_unique_candidates": 0,
    "uri_ambiguous_candidates": 0,
    "uri_unmatched": 0
}

for identity in identities:

    internal_id = identity["movie_entity_id"]

    identity_type = identity["identity_type"]

    canonical_id = identity["canonical_identifier"]

    is_numeric = is_numeric_identity(identity_type)

    result = {
        "movie_entity_id": internal_id,
        "identity_type": identity_type,
        "canonical_identifier": canonical_id,
        "movie_title": "",
        "reference_movie_id": "",
        "reference_mention_count": "",
        "uri_local_label": "",
        "candidate_count": 0,
        "metadata_source": "",
        "title_match_status": "",
        "identity_status": identity.get(
            "identity_status", ""
        )
    }

    # --------------------------------------------------------
    # Numeric identities
    # --------------------------------------------------------

    if is_numeric:

        counts["numeric"] += 1

        external_id = str(canonical_id).strip()

        reference = reference_by_id.get(
            external_id
        )

        require(
            reference is not None,
            f"Missing numeric reference: {external_id}"
        )

        require(
            numeric_verified.get(external_id, False),
            f"Numeric identity not independently verified: {external_id}"
        )

        result["movie_title"] = reference["movieName"]

        result["reference_movie_id"] = external_id

        result["reference_mention_count"] = reference["nbMentions"]

        result["metadata_source"] = (
            "movies_with_mentions.csv; "
            "Experiment 21.14 semantic crosswalk"
        )

        result["title_match_status"] = (
            "VERIFIED_NUMERIC_CROSSWALK"
        )

        counts["numeric_verified"] += 1

    # --------------------------------------------------------
    # DBpedia URI identities
    # --------------------------------------------------------

    else:

        counts["uri"] += 1

        label = uri_local_label(canonical_id)

        result["uri_local_label"] = label

        label_key = normalize_title(label)

        candidates = title_index.get(
            label_key,
            []
        ) if label_key else []

        result["candidate_count"] = len(candidates)

        if len(candidates) == 1:

            candidate = candidates[0]

            result["movie_title"] = candidate["movieName"]

            result["reference_movie_id"] = candidate["movieId"]

            result["reference_mention_count"] = candidate["nbMentions"]

            result["metadata_source"] = (
                "movies_with_mentions.csv; "
                "exact normalized URI-label candidate"
            )

            result["title_match_status"] = (
                "CANDIDATE_EXACT_MATCH_NOT_VERIFIED"
            )

            counts["uri_unique_candidates"] += 1

        elif len(candidates) > 1:

            result["title_match_status"] = (
                "AMBIGUOUS_TITLE_CANDIDATE"
            )

            counts["uri_ambiguous_candidates"] += 1

        else:

            result["title_match_status"] = (
                "NO_EXACT_TITLE_CANDIDATE"
            )

            counts["uri_unmatched"] += 1

        uri_candidates.append(result.copy())

    complete_inventory.append(result)


# ============================================================
# 7. STRICT POPULATION VALIDATION
# ============================================================

print("\n[4] POPULATION VALIDATION")

require(
    len(complete_inventory) == 6924,
    "Final inventory population mismatch."
)

require(
    counts["numeric"] == 812,
    "Numeric population mismatch."
)

require(
    counts["uri"] == 6112,
    "URI population mismatch."
)

require(
    counts["numeric_verified"] == 812,
    "Numeric verification mismatch."
)

require(
    len({
        row["movie_entity_id"]
        for row in complete_inventory
    }) == 6924,
    "Duplicate internal IDs in final inventory."
)

print("Total identities:", len(complete_inventory))
print("Numeric identities:", counts["numeric"])
print("URI identities:", counts["uri"])
print("Numeric verified:", counts["numeric_verified"])

print("\nURI TITLE CANDIDATES")

print(
    "Unique exact candidates:",
    counts["uri_unique_candidates"]
)

print(
    "Ambiguous candidates:",
    counts["uri_ambiguous_candidates"]
)

print(
    "Unmatched:",
    counts["uri_unmatched"]
)


# ============================================================
# 8. EXPORT RESULTS
# ============================================================

print("\n[5] EXPORTING RESULTS")

inventory_fields = list(
    complete_inventory[0].keys()
)

write_csv(
    OUTPUT / "movie_metadata_enrichment_inventory.csv",
    complete_inventory,
    inventory_fields
)

write_csv(
    OUTPUT / "uri_title_candidate_audit.csv",
    uri_candidates,
    inventory_fields
)

# Create validation manifest.

manifest = {
    "experiment": "21.16A",
    "name": "URI Movie Title Candidate and Metadata Source Audit",

    "purpose": (
        "Audit title candidates for DBpedia URI entities "
        "and reconcile verified numeric movie identities."
    ),

    "source_files": [
        str(IDENTITY_FILE),
        str(MENTION_FILE),
        str(NUMERIC_ALIGNMENT_FILE)
    ],

    "counts": {
        "total_movie_entities": len(complete_inventory),
        **counts
    },

    "validation": {
        "catalog_population": len(complete_inventory) == 6924,
        "internal_id_uniqueness": len(set(internal_ids)) == 6924,
        "reference_id_uniqueness": len(set(reference_ids)) == 6924,
        "numeric_population": counts["numeric"] == 812,
        "uri_population": counts["uri"] == 6112,
        "numeric_crosswalk_verification": counts["numeric_verified"] == 812,
        "inventory_population": len(complete_inventory) == 6924
    },

    "interpretation": {
        "numeric_titles": "Verified using Experiment 21.14.",
        "uri_exact_matches": (
            "Candidate matches only; require independent validation."
        ),
        "ambiguous_titles": "Require further investigation.",
        "unmatched_titles": "Require additional metadata sources.",
        "uri_local_labels": (
            "Never treated as independently verified movie titles."
        )
    },

    "limitations": [
        "An exact normalized URI-label match does not prove movie identity.",
        "Similar titles may refer to different films.",
        "No external metadata retrieval was performed.",
        "No genre classification was performed."
    ],

    "outputs": [
        "movie_metadata_enrichment_inventory.csv",
        "uri_title_candidate_audit.csv",
        "experiment_21_16A_manifest.json"
    ]
}

require(
    all(manifest["validation"].values()),
    "One or more strict validation criteria failed."
)

with (
    OUTPUT / "experiment_21_16A_manifest.json"
).open("w", encoding="utf-8") as f:

    json.dump(
        manifest,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 75)
print("EXPERIMENT 21.16A COMPLETED")
print("=" * 75)

print("Output directory:", OUTPUT.resolve())

for filename in manifest["outputs"]:

    path = OUTPUT / filename

    print(
        filename,
        "|",
        path.stat().st_size,
        "bytes"
    )

print("\nOriginal source files unchanged.")
print("No graph modification.")
print("No model training or inference.")
print("URI candidate matches have NOT been declared verified.")

print("\nEXPERIMENT 21.16A FINISHED.")
