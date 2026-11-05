#!/usr/bin/env python
"""Experiment 21.16H: conservative record-level movie identity evidence reconstruction.

Run from repository root:
  python -m py_compile .\21_16H_record_level_movie_identity_reconstruction.py
  python .\21_16H_record_level_movie_identity_reconstruction.py

No source files, graph, or model files are modified. No automatic merging occurs.
"""
from __future__ import annotations
import csv, hashlib, json, re, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "results_category_audit"
REDIAL = ROOT / "external_validation" / "redial_reference"
ML32 = ROOT / "external_validation" / "movielens_32m"
OUT = AUDIT / "experiment_21_16H"

# Each source is loaded from the experiment that actually generated it.
# Experiment 21.16G contains the schema inventory only; it is not the
# storage location for the underlying audit CSV files.
FILES = {
    "identity": AUDIT / "experiment_21_15" / "unified_movie_entity_mapping.csv",
    "pairwise": AUDIT / "experiment_21_16E" / "movielens_conflict_pairwise_analysis.csv",
    "conflict_summary": AUDIT / "experiment_21_16F" / "external_identifier_group_summary.csv",
    "uri_audit": AUDIT / "experiment_21_16A" / "uri_title_candidate_audit.csv",
    "merged": REDIAL / "movies_merged.csv",
    "mentions": REDIAL / "movies_with_mentions.csv",
    "links": ML32 / "links.csv",
    "movies": ML32 / "movies.csv",
}
N_ID, N_URI, N_NUM, N_PAIR, N_GROUP = 6924, 6112, 812, 104, 100


def say(x=""):
    print(x, flush=True)
def json_default(obj):
    """
    Convert NumPy and Pandas scalar values
    into native Python values for JSON serialization.
    """
    if hasattr(obj, "item"):
        return obj.item()

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )

def clean(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    if re.fullmatch(r"-?\d+\.0+", x):
        x = x.split(".", 1)[0]
    return x


def read(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig", low_memory=False)


def norm_title(x):
    # Conservative: does not strip years, articles, sequel numbers or qualifiers.
    x = unquote(clean(x)).casefold().replace("&", " and ")
    x = x.replace("_", " ")
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    return re.sub(r"\s+", " ", x).strip()


def uri_label(x):
    x = clean(x).rstrip("/")
    return unquote(x.rsplit("/", 1)[-1]).replace("_", " ") if x else ""


def unique_json(series):
    vals, seen = [], set()
    for x in series:
        x = clean(x)
        if x and x not in seen:
            vals.append(x)
            seen.add(x)
    return json.dumps(vals, ensure_ascii=False)


def first_value(series):
    for x in series:
        x = clean(x)
        if x:
            return x
    return ""


def check_cols(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing {missing}; found {df.columns.tolist()}")


def write_csv(df, name):
    p = OUT / name
    df.to_csv(p, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    say(f"{name}: {len(df):,} rows, {p.stat().st_size:,} bytes")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources():
    say("\n[1] SOURCE INVENTORY")
    s = {}
    for k in ["identity", "pairwise", "merged", "mentions", "links", "movies"]:
        s[k] = read(FILES[k])
        say(f"{k}: {len(s[k]):,} rows")
    for k in ["uri_audit", "conflict_summary"]:
        s[k] = read(FILES[k]) if FILES[k].exists() else pd.DataFrame()
        say(f"{k}: {len(s[k]):,} rows" if not s[k].empty else f"{k}: unavailable (optional)")
    return s


def validate_sources(s):
    say("\n[2] SOURCE SCHEMA AND INTEGRITY")
    check_cols(s["identity"], ["movie_entity_id", "canonical_entity_id",
        "identity_type", "canonical_identifier", "identity_provenance", "identity_status"], "identity")
    check_cols(s["pairwise"], ["movielens_id", "database_id_a", "database_id_b"], "pairwise")
    check_cols(s["merged"], ["movieName", "databaseId", "movielensId"], "merged")
    check_cols(s["mentions"], ["movieId", "movieName", "nbMentions"], "mentions")
    check_cols(s["links"], ["movieId", "imdbId", "tmdbId"], "links")
    check_cols(s["movies"], ["movieId", "title", "genres"], "movies")

    for df, cols in [
        (s["identity"], ["movie_entity_id", "canonical_identifier"]),
        (s["pairwise"], ["movielens_id", "database_id_a", "database_id_b"]),
        (s["merged"], ["databaseId", "movielensId"]),
        (s["mentions"], ["movieId"]), (s["links"], ["movieId"]), (s["movies"], ["movieId"])
    ]:
        for c in cols:
            df[c] = df[c].map(clean)

    ident = s["identity"]
    # identity_type is a column of the identity DataFrame, not a top-level
    # key in the source dictionary.
    typ = ident["identity_type"]
    checks = {
        "identity_population_6924": len(ident) == N_ID,
        "internal_entity_ids_unique": ident.movie_entity_id.is_unique,
        "canonical_ids_nonblank": ident.canonical_identifier.ne("").all(),
        "canonical_ids_unique": ident.canonical_identifier.is_unique,
        "URI_population_6112": int(typ.eq("DBpedia_URI").sum()) == N_URI,
        "numeric_population_812": int(typ.eq("Numeric_External_ID").sum()) == N_NUM,
        "mention_ids_unique": s["mentions"].movieId.is_unique,
        "MovieLens_link_ids_unique": s["links"].movieId.is_unique,
        "MovieLens_movie_ids_unique": s["movies"].movieId.is_unique,
        "pairwise_population_104": len(s["pairwise"]) == N_PAIR,
        "merged_ids_nonblank": s["merged"].databaseId.ne("").all() and s["merged"].movielensId.ne("").all(),
    }
    for k, v in checks.items():
        say(f"{k}: {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise ValueError("Source validation failed: " + json.dumps(checks))
    return checks


def build_db_inventory(s):
    say("\n[3] DATABASE-ID RECORD INVENTORY")
    m = s["merged"].copy()
    q = s["mentions"].copy()
    m["norm_title"] = m.movieName.map(norm_title)
    q["norm_mention"] = q.movieName.map(norm_title)

    db = m.groupby("databaseId").agg(
        merged_source_row_count=("databaseId", "size"),
        merged_titles=("movieName", unique_json),
        normalized_merged_titles=("norm_title", unique_json),
        movielens_ids=("movielensId", unique_json),
        distinct_merged_title_count=("norm_title", "nunique"),
        distinct_movielens_id_count=("movielensId", "nunique"),
    ).reset_index().rename(columns={"databaseId": "database_id"})

    q = q[["movieId", "movieName", "nbMentions", "norm_mention"]].rename(columns={
        "movieId": "database_id", "movieName": "mention_title",
        "nbMentions": "mention_count", "norm_mention": "normalized_mention_title"})
    db = db.merge(q, on="database_id", how="left", validate="one_to_one")
    db["mention_reference_found"] = db.mention_title.notna() & db.mention_title.ne("")
    db["mention_title_exact_match"] = db.apply(
        lambda r: bool(r.normalized_mention_title) and
        r.normalized_mention_title in set(json.loads(r.normalized_merged_titles)), axis=1)

    rel = m[["databaseId", "movielensId", "movieName"]].drop_duplicates().rename(columns={
        "databaseId": "database_id", "movielensId": "movielens_id"})
    rel = rel.merge(s["links"].rename(columns={"movieId": "movielens_id"}),
                    on="movielens_id", how="left", validate="many_to_one")
    rel = rel.merge(s["movies"].rename(columns={"movieId": "movielens_id",
                    "title": "movielens_reference_title", "genres": "movielens_reference_genres"}),
                    on="movielens_id", how="left", validate="many_to_one")
    rel["database_title_normalized"] = rel.movieName.map(norm_title)
    rel["movielens_title_normalized"] = rel.movielens_reference_title.map(norm_title)
    rel["database_title_matches_movielens_title"] = (
        rel.database_title_normalized.ne("") &
        rel.database_title_normalized.eq(rel.movielens_title_normalized))
    rel["links_reference_found"] = rel.imdbId.map(lambda x: clean(x) not in {"", "0", "-1"})
    rel["movie_metadata_found"] = rel.movielens_reference_title.notna() & rel.movielens_reference_title.ne("")

    agg = rel.groupby("database_id").agg(
        relation_row_count=("movielens_id", "size"),
        external_movielens_ids=("movielens_id", unique_json),
        imdb_ids=("imdbId", unique_json), tmdb_ids=("tmdbId", unique_json),
        movielens_reference_titles=("movielens_reference_title", unique_json),
        movielens_reference_genres=("movielens_reference_genres", unique_json),
        external_link_rows_found=("links_reference_found", "sum"),
        external_movie_metadata_rows_found=("movie_metadata_found", "sum"),
        database_to_movielens_title_match_rows=("database_title_matches_movielens_title", "sum"),
    ).reset_index()
    db = db.merge(agg, on="database_id", how="left", validate="one_to_one")
    for c in ["relation_row_count", "external_link_rows_found",
              "external_movie_metadata_rows_found", "database_to_movielens_title_match_rows"]:
        db[c] = db[c].fillna(0).astype(int)
    db["record_identity_decision"] = "UNRESOLVED"
    db["record_decision_basis"] = "Evidence inventory only; no automatic identity decision."
    db["external_identifier_scope"] = (
        "IMDb/TMDb values are reached through MovieLens IDs; they are not direct "
        "databaseId-level verification."
    )
    say(f"Distinct database IDs: {len(db):,}")
    say(f"Database IDs with mention references: {int(db.mention_reference_found.sum()):,}")
    say(f"Database IDs with at least one linked IMDb ID: {int((db.external_link_rows_found > 0).sum()):,}")
    return db, rel


def build_conflicts(s, db, rel):
    say("\n[4] PAIRWISE CONFLICT RECONSTRUCTION")
    p = s["pairwise"].copy()
    da = db.add_prefix("a_").rename(columns={"a_database_id": "database_id_a"})
    dbb = db.add_prefix("b_").rename(columns={"b_database_id": "database_id_b"})
    out = p.merge(da, on="database_id_a", how="left", validate="many_to_one")
    out = out.merge(dbb, on="database_id_b", how="left", validate="many_to_one")

    relation_set = set(zip(rel.database_id.map(clean), rel.movielens_id.map(clean)))
    out["database_a_linked_to_pair_movielens_id"] = [
        (a, ml) in relation_set for a, ml in zip(out.database_id_a, out.movielens_id)]
    out["database_b_linked_to_pair_movielens_id"] = [
        (b, ml) in relation_set for b, ml in zip(out.database_id_b, out.movielens_id)]
    out["pair_reference_membership_valid"] = (
        out.database_a_linked_to_pair_movielens_id &
        out.database_b_linked_to_pair_movielens_id)

    links = s["links"].rename(columns={"movieId": "movielens_id",
        "imdbId": "group_imdb_id", "tmdbId": "group_tmdb_id"})
    movies = s["movies"].rename(columns={"movieId": "movielens_id",
        "title": "group_movielens_title", "genres": "group_movielens_genres"})
    out = out.merge(links, on="movielens_id", how="left", validate="many_to_one")
    out = out.merge(movies, on="movielens_id", how="left", validate="many_to_one")
    out["group_imdb_available"] = out.group_imdb_id.map(lambda x: clean(x) not in {"", "0", "-1"})
    out["group_tmdb_available"] = out.group_tmdb_id.map(lambda x: clean(x) not in {"", "0", "-1"})
    out["external_evidence_scope"] = "GROUP_LEVEL_ONLY: identifiers belong to shared MovieLens ID."
    out["pair_identity_decision"] = "UNRESOLVED"
    out["automatic_merge"] = False
    out["decision_reason"] = (
        "Shared MovieLens ID, title agreement and group-level external IDs are candidate "
        "evidence only; no direct databaseId-to-IMDb/TMDb linkage is established."
    )

    groups = out.groupby("movielens_id").agg(
        pairwise_comparisons=("movielens_id", "size"),
        unique_database_ids=("database_id_a", "nunique"),
        database_ids_a=("database_id_a", unique_json),
        database_ids_b=("database_id_b", unique_json),
        valid_pair_membership_rows=("pair_reference_membership_valid", "sum"),
        imdb_id=("group_imdb_id", first_value),
        tmdb_id=("group_tmdb_id", first_value),
        movielens_title=("group_movielens_title", first_value),
    ).reset_index()
    groups["group_identity_decision"] = "UNRESOLVED"
    groups["decision_scope"] = "Conflict-group evidence; no automatic entity merging."
    say(f"Pairwise records: {len(out):,}")
    say(f"Conflict groups: {len(groups):,}")
    say(f"Pairwise records with valid source membership: {int(out.pair_reference_membership_valid.sum()):,}")
    return out, groups


def build_uri(s):
    say("\n[5] URI IDENTITY EVIDENCE")
    u = s["identity"][s["identity"]["identity_type"].eq("DBpedia_URI")].copy()
    if "uri_local_name" not in u:
        u["uri_local_name"] = ""
    u["uri_local_label_derived"] = u.apply(
        lambda r: clean(r.uri_local_name) or uri_label(r.canonical_identifier), axis=1)
    u["uri_local_label_normalized"] = u.uri_local_label_derived.map(norm_title)
    audit = s["uri_audit"]
    if not audit.empty and "movie_entity_id" in audit:
        audit = audit.copy()
        audit.movie_entity_id = audit.movie_entity_id.map(clean)
        audit = audit.drop_duplicates("movie_entity_id")
        audit = audit.rename(columns={c: "prior_uri_audit_" + c
                            for c in audit.columns if c != "movie_entity_id"})
        u = u.merge(audit, on="movie_entity_id", how="left", validate="one_to_one")
        audit_cols = [c for c in u if c.startswith("prior_uri_audit_")]
        u["prior_candidate_audit_found"] = u[audit_cols].notna().any(axis=1)
    else:
        u["prior_candidate_audit_found"] = False
    u["uri_identity_decision"] = "UNRESOLVED"
    u["uri_candidate_is_identity_proof"] = False
    u["uri_evidence_interpretation"] = (
        "Original URI preserved; local label/title candidate matches are not independent identity proof."
    )
    say(f"URI identities: {len(u):,}")
    say(f"URI rows with prior candidate audit: {int(u.prior_candidate_audit_found.sum()):,}")
    return u


def build_numeric(s):
    say("\n[6] NUMERIC IDENTITY EVIDENCE")
    n = s["identity"][s["identity"]["identity_type"].eq("Numeric_External_ID")].copy()
    for c in ["external_numeric_id", "mapping_value"]:
        if c not in n:
            n[c] = ""
    n["numeric_external_identifier"] = n.apply(
        lambda r: clean(r.external_numeric_id) or clean(r.mapping_value) or
                  clean(r.canonical_identifier), axis=1)
    n["numeric_identity_decision"] = n.identity_status.map(
        lambda x: "VERIFIED_SAME_ENTITY" if clean(x).upper() == "VERIFIED" else "UNRESOLVED")
    n["numeric_verification_scope"] = (
        "Prior Experiment 21.15 status carried forward. 21.16H does not rerun 21.14."
    )
    n["identifier_semantics"] = (
        "Numeric identifier preserved; original generating namespace/process not inferred."
    )
    say(f"Numeric identities: {len(n):,}")
    say(f"Prior VERIFIED status carried forward: {int(n.numeric_identity_decision.eq('VERIFIED_SAME_ENTITY').sum()):,}")
    return n


def build_master(s, db):
    say("\n[7] MASTER IDENTITY TABLE AND DECISION REGISTER")
    d = db.rename(columns={"database_id": "movie_entity_id"})
    master = s["identity"].merge(d, on="movie_entity_id", how="left", validate="one_to_one")
    master["database_record_evidence_found"] = master.merged_source_row_count.notna()
    master["automatic_merge"] = False
    master["experiment_21_16H_identity_decision"] = "UNRESOLVED"
    master["decision_scope"] = "Evidence reconstruction; no new merge or identity upgrade."

    # Carry forward only the existing VERIFIED status for the 812 numeric IDs.
    mask = master.identity_type.eq("Numeric_External_ID") & master.identity_status.str.upper().eq("VERIFIED")
    master.loc[mask, "experiment_21_16H_identity_decision"] = "VERIFIED_SAME_ENTITY"
    master.loc[mask, "decision_scope"] = (
        "Prior validated numeric crosswalk status carried forward; not newly verified by 21.16H.")
    # All URI-based identities deliberately remain unresolved.
    master.loc[master.identity_type.eq("DBpedia_URI"),
               "experiment_21_16H_identity_decision"] = "UNRESOLVED"

    cols = [c for c in [
        "movie_entity_id", "canonical_entity_id", "identity_type", "canonical_identifier",
        "identity_provenance", "identity_status", "database_record_evidence_found",
        "mention_reference_found", "mention_title_exact_match", "distinct_merged_title_count",
        "distinct_movielens_id_count", "imdb_ids", "tmdb_ids",
        "experiment_21_16H_identity_decision", "decision_scope", "automatic_merge"
    ] if c in master.columns]
    decisions = master[cols].copy()
    unresolved = decisions[decisions.experiment_21_16H_identity_decision.eq("UNRESOLVED")].copy()
    unresolved["unresolved_reason"] = unresolved.identity_type.map({
        "DBpedia_URI": "Source URI preserved, but independent record-level movie identity is not established.",
        "Numeric_External_ID": "Prior identity status was not VERIFIED; manual/source review required."
    })
    say(f"Master rows: {len(master):,}; unresolved: {len(unresolved):,}")
    return master, decisions, unresolved


def write_outputs(s, checks, db, rel, pair, groups, uri, numeric, master, decisions, unresolved):
    say("\n[8] FINAL VALIDATION")
    validation = {
        "master_population_6924": len(master) == N_ID,
        "master_internal_ids_unique": master.movie_entity_id.is_unique,
        "master_canonical_ids_unique": master.canonical_identifier.is_unique,
        "uri_population_6112": len(uri) == N_URI,
        "numeric_population_812": len(numeric) == N_NUM,
        "pairwise_population_104": len(pair) == N_PAIR,
        "conflict_groups_100": len(groups) == N_GROUP,
        "decision_register_6924": len(decisions) == N_ID,
        "all_uri_decisions_unresolved": master.loc[
            master.identity_type.eq("DBpedia_URI"),
            "experiment_21_16H_identity_decision"].eq("UNRESOLVED").all(),
        "no_automatic_merges": not master.automatic_merge.any(),
        "database_inventory_nonempty": len(db) > 0,
        "database_movielens_relations_nonempty": len(rel) > 0,
    }
    for k, v in validation.items():
        say(f"{k}: {'PASS' if v else 'FAIL'}")
    if not all(validation.values()):
        raise ValueError("Final validation failed: " + json.dumps(validation))

    say("\n[9] EXPORTING")
    tables = {
        "movie_identity_evidence_master.csv": master,
        "database_id_record_evidence.csv": db,
        "database_id_movielens_relationships.csv": rel,
        "conflict_record_reconstruction.csv": pair,
        "conflict_group_reconstruction.csv": groups,
        "uri_identity_reconstruction.csv": uri,
        "numeric_identity_reconstruction.csv": numeric,
        "identity_decision_register.csv": decisions,
        "unresolved_identity_records.csv": unresolved,
    }
    for name, df in tables.items():
        write_csv(df, name)

    file_info = {}
    for key, path in FILES.items():
        file_info[key] = ({"path": str(path.relative_to(ROOT)),
                           "bytes": path.stat().st_size, "sha256": sha256(path)}
                          if path.exists() else {"status": "NOT_AVAILABLE"})
    manifest = {
        "experiment": "21.16H",
        "name": "Record-Level Movie Identity Reconstruction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reconstruct evidence for all 6,924 identities without automatic merging.",
        "source_files": file_info,
        "source_validation": checks,
        "counts": {
            "master_identities": len(master), "uri_identities": len(uri),
            "numeric_identities": len(numeric), "database_ids": len(db),
            "database_movielens_relationships": len(rel), "pairwise_records": len(pair),
            "conflict_groups": len(groups), "unresolved_identities": len(unresolved),
        },
        "decision_distribution": {str(k): int(v) for k, v in
            master.experiment_21_16H_identity_decision.value_counts().items()},
        "validation": validation,
        "policy": {
            "automatic_merging": False,
            "title_match_is_identity_proof": False,
            "URI_label_is_authoritative_title": False,
            "shared_MovieLens_ID_is_individual_identity_proof": False,
            "IMDb_TMDb_scope": "MovieLens-group-level cross-reference only.",
            "numeric_status": "Prior 21.15 status carried forward; no fresh 21.14 rerun.",
            "uncertain_relationships": "Remain UNRESOLVED.",
            "original_sources_modified": False, "graph_modified": False,
            "model_training": False, "model_inference": False,
        },
        "limitations": [
            "MovieLens links associates IMDb/TMDb IDs with MovieLens IDs, not directly with each ReDial databaseId.",
            "A shared MovieLens ID does not independently prove two databaseId records are the same real-world movie.",
            "Title/year agreement is candidate evidence only.",
            "URI local labels and prior URI title candidates are not independent identity verification.",
            "This script does not query IMDb or TMDb directly.",
            "Numeric verification status is carried forward from the earlier workflow.",
        ],
        "outputs": list(tables) + ["experiment_21_16H_manifest.json",
                                   "experiment_21_16H_validation_report.txt"],
        "completion_status": "COMPLETED_VALIDATIONS_PASSED_WITH_UNRESOLVED_IDENTITIES",
    }
    mp = OUT / "experiment_21_16H_manifest.json"
    mp.write_text(
    json.dumps(
        manifest,
        indent=4,
        ensure_ascii=False,
        default=json_default
    ),
    encoding="utf-8"
    )
    report = [
        "EXPERIMENT 21.16H - VALIDATION REPORT", "=" * 70,
        f"UTC: {manifest['created_utc']}", "",
        "COUNTS",
    ] + [f"{k}: {v}" for k, v in manifest["counts"].items()] + [
        "", "DECISION DISTRIBUTION"
    ] + [f"{k}: {v}" for k, v in manifest["decision_distribution"].items()] + [
        "", "VALIDATION"
    ] + [f"{k}: {'PASS' if v else 'FAIL'}" for k, v in validation.items()] + [
        "", "SAFEGUARDS",
        "- No automatic entity merging.",
        "- URI candidates are not treated as identity proof.",
        "- IMDb/TMDb identifiers are group-level evidence only.",
        "- Uncertain relationships remain unresolved.",
        "- Original data, graph and model files are not modified.",
    ]
    (OUT / "experiment_21_16H_validation_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    say(f"\nManifest: {mp}")
    say(f"Output directory: {OUT}")


def main():
    say("=" * 75)
    say("EXPERIMENT 21.16H: RECORD-LEVEL MOVIE IDENTITY RECONSTRUCTION")
    say("=" * 75)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        s = load_sources()
        checks = validate_sources(s)
        db, rel = build_db_inventory(s)
        pair, groups = build_conflicts(s, db, rel)
        uri = build_uri(s)
        numeric = build_numeric(s)
        master, decisions, unresolved = build_master(s, db)
        write_outputs(s, checks, db, rel, pair, groups, uri, numeric,
                      master, decisions, unresolved)
        say("\n" + "=" * 75)
        say("EXPERIMENT 21.16H COMPLETED")
        say(f"Identities: {len(master):,}/{N_ID:,}")
        say(f"URI: {len(uri):,}; numeric: {len(numeric):,}")
        say(f"Pairwise conflicts: {len(pair):,}/{N_PAIR:,}; groups: {len(groups):,}/{N_GROUP:,}")
        say(f"Unresolved identities: {len(unresolved):,}")
        say("No automatic merging; no graph modification; no model training/inference.")
        say("EXPERIMENT 21.16H FINISHED.")
        return 0
    except Exception as e:
        say("\nEXPERIMENT 21.16H FAILED")
        say(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        say("Do not interpret partial outputs as completed results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
