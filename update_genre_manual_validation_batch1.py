"""
Apply the reviewed Batch 1 labels to the DBpedia manual-validation CSV.

Run from the project root:
    python update_genre_manual_validation_batch1.py

Default input:
    results_category_audit/dbpedia_pilot/genre_manual_validation.csv

Output:
    results_category_audit/dbpedia_pilot/genre_manual_validation_batch1_updated.csv

The source file is never overwritten.
"""
from pathlib import Path
import pandas as pd

INPUT = Path("results_category_audit/dbpedia_pilot/genre_manual_validation.csv")
OUTPUT = Path("results_category_audit/dbpedia_pilot/genre_manual_validation_batch1_updated.csv")

# category: (validated_type, canonical_genre, parent_genre,
#            temporal_modifier, geographical_modifier, manual_decision, review_notes)
BATCH1 = {'1923 drama films': ('primary_genre', 'Drama', '', '1923', '', 'ACCEPT_GENRE', 'Genre label denotes a recognized film genre; year retained separately.'), '1940 drama films': ('primary_genre', 'Drama', '', '1940', '', 'ACCEPT_GENRE', 'Genre label denotes a recognized film genre; year retained separately.'), '1950s Western (genre) musical films': ('hybrid_genre', 'Western;Musical', '', '1950s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '1955 musical films': ('primary_genre', 'Musical', '', '1955', '', 'ACCEPT_GENRE', 'Musical is treated as a genre; year retained separately.'), '1965 Western (genre) films': ('primary_genre', 'Western', '', '1965', '', 'ACCEPT_GENRE', 'Western is treated as a genre; year retained separately.'), '1980s superhero films': ('subgenre', 'Superhero', 'Action', '', '', 'ACCEPT_GENRE', 'Superhero is retained as a genre/subgenre label; decade retained separately.'), '1990s adventure drama films': ('hybrid_genre', 'Adventure;Drama', '', '1990s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), "1990s children's adventure films": ('primary_genre', 'Adventure', '', '1990s', 'Children', 'ACCEPT_GENRE', 'Adventure is genre; audience is a separate modifier.'), "1990s children's animated films": ('format_attribute', 'Animation', '', '1990s', 'Children', 'ATTRIBUTE_ONLY', 'Animation is a format, not a genre; audience is separate.'), "1990s children's comedy films": ('primary_genre', 'Comedy', '', '1990s', 'Children', 'ACCEPT_GENRE', 'Comedy is genre; audience is a separate modifier.'), "1990s children's drama films": ('primary_genre', 'Drama', '', '1990s', 'Children', 'ACCEPT_GENRE', 'Drama is genre; audience is a separate modifier.'), "1990s children's fantasy films": ('primary_genre', 'Fantasy', '', '1990s', 'Children', 'ACCEPT_GENRE', 'Fantasy is genre; audience is a separate modifier.'), '1990s fantasy comedy films': ('hybrid_genre', 'Fantasy;Comedy', '', '1990s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '1994 drama films': ('primary_genre', 'Drama', '', '1994', '', 'ACCEPT_GENRE', 'Drama is genre; year retained separately.'), '1994 romantic drama films': ('hybrid_genre', 'Romance;Drama', '', '1994', '', 'ACCEPT_HYBRID', 'Romance and drama are retained as separate genre components.'), '1996 comedy films': ('primary_genre', 'Comedy', '', '1996', '', 'ACCEPT_GENRE', 'Comedy is genre; year retained separately.'), '1998 American animated films': ('format_attribute', 'Animation', '', '1998', 'United States', 'ATTRIBUTE_ONLY', 'Animation is a format; country and year are separate modifiers.'), '1998 comedy films': ('primary_genre', 'Comedy', '', '1998', '', 'ACCEPT_GENRE', 'Comedy is genre; year retained separately.'), '1998 computer-animated films': ('format_attribute', 'Computer animation', 'Animation', '1998', '', 'ATTRIBUTE_ONLY', 'Computer animation is a production/visual format, not a genre.'), '2000s action adventure films': ('hybrid_genre', 'Action;Adventure', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s action horror films': ('hybrid_genre', 'Action;Horror', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s adventure comedy films': ('hybrid_genre', 'Adventure;Comedy', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s buddy comedy films': ('subgenre', 'Buddy comedy', 'Comedy', '2000s', '', 'ACCEPT_GENRE', 'Buddy comedy treated as a comedy subgenre.'), "2000s children's adventure films": ('primary_genre', 'Adventure', '', '2000s', 'Children', 'ACCEPT_GENRE', 'Adventure is genre; audience is a separate modifier.'), "2000s children's animated films": ('format_attribute', 'Animation', '', '2000s', 'Children', 'ATTRIBUTE_ONLY', 'Animation is a format, not a genre; audience is separate.'), "2000s children's comedy films": ('primary_genre', 'Comedy', '', '2000s', 'Children', 'ACCEPT_GENRE', 'Comedy is genre; audience is a separate modifier.'), '2000s comedy road movies': ('hybrid_genre', 'Comedy;Road movie', '', '2000s', '', 'ACCEPT_HYBRID', 'Comedy and road movie are retained as genre/narrative-type components.'), '2000s coming-of-age drama films': ('hybrid_genre', 'Coming-of-age;Drama', '', '2000s', '', 'ACCEPT_HYBRID', 'Coming-of-age narrative and drama genre are retained separately.'), '2000s coming-of-age films': ('narrative_subgenre', 'Coming-of-age', '', '2000s', '', 'ACCEPT_GENRE', 'Coming-of-age is treated as a narrative genre/subgenre.'), '2000s fantasy adventure films': ('hybrid_genre', 'Fantasy;Adventure', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s fantasy comedy films': ('hybrid_genre', 'Fantasy;Comedy', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s romance films': ('primary_genre', 'Romance', '', '2000s', '', 'ACCEPT_GENRE', 'Romance is genre; decade retained separately.'), '2000s science fiction adventure films': ('hybrid_genre', 'Science fiction;Adventure', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s science fiction comedy films': ('hybrid_genre', 'Science fiction;Comedy', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2000s science fiction war films': ('hybrid_genre', 'Science fiction;War', '', '2000s', '', 'ACCEPT_HYBRID', 'Two genre components; decade retained separately.'), '2004 black comedy films': ('subgenre', 'Black comedy', 'Comedy', '2004', '', 'ACCEPT_GENRE', 'Black comedy is treated as a comedy subgenre.'), '2004 comedy films': ('primary_genre', 'Comedy', '', '2004', '', 'ACCEPT_GENRE', 'Comedy is genre; year retained separately.'), '2004 science fiction films': ('primary_genre', 'Science fiction', '', '2004', '', 'ACCEPT_GENRE', 'Science fiction is genre; year retained separately.'), '2007 American animated films': ('format_attribute', 'Animation', '', '2007', 'United States', 'ATTRIBUTE_ONLY', 'Animation is a format; country and year are separate modifiers.'), '2007 action thriller films': ('hybrid_genre', 'Action;Thriller', '', '2007', '', 'ACCEPT_HYBRID', 'Two genre components; year retained separately.')}

REQUIRED_COLUMNS = [
    "original_category", "value_uri_or_literal", "preliminary_class",
    "validated_type", "canonical_genre", "parent_genre",
    "temporal_modifier", "geographical_modifier",
    "manual_decision", "review_notes",
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}\\n"
            "Run this script from your project root, or edit INPUT at the top."
        )

    df = pd.read_csv(INPUT, encoding="utf-8-sig")

    # Normalize common header variations from spreadsheet/Markdown exports.
    df.columns = [
        str(c).replace("**", "").replace("&#x72;", "r").strip()
        for c in df.columns
    ]
    rename = {"**manual_decision": "manual_decision",
              "**review_notes": "review_notes"}
    df = df.rename(columns=rename)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["original_category"] = df["original_category"].astype(str).str.strip()

    missing = sorted(set(BATCH1) - set(df["original_category"]))
    if missing:
        raise ValueError("These Batch 1 categories were not found: " + repr(missing))

    duplicate_mask = df["original_category"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicated = sorted(df.loc[duplicate_mask, "original_category"].unique())
        raise ValueError("Duplicate category labels found; resolve before updating: " + repr(duplicated))

    for category, values in BATCH1.items():
        (validated_type, canonical_genre, parent_genre,
         temporal_modifier, geographical_modifier,
         manual_decision, review_notes) = values

        mask = df["original_category"].eq(category)
        df.loc[mask, [
            "validated_type", "canonical_genre", "parent_genre",
            "temporal_modifier", "geographical_modifier",
            "manual_decision", "review_notes"
        ]] = [
            validated_type, canonical_genre, parent_genre,
            temporal_modifier, geographical_modifier,
            manual_decision, review_notes
        ]

    # Keep original columns first, then any extra columns.
    ordered = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[ordered]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    reviewed = df[df["manual_decision"].isin(
        ["ACCEPT_GENRE", "ACCEPT_HYBRID", "ATTRIBUTE_ONLY", "REVIEW"]
    )]
    print("=== BATCH 1 UPDATE COMPLETE ===")
    print("Input :", INPUT)
    print("Output:", OUTPUT)
    print("Total category records:", len(df))
    print("Batch 1 categories updated:", len(BATCH1))
    print("\nDecision counts:")
    print(reviewed["manual_decision"].value_counts().to_string())
    print("\nUpdated category examples:")
    print(reviewed[[
        "original_category", "validated_type",
        "canonical_genre", "manual_decision"
    ]].head(40).to_string(index=False))
    print("\nOriginal input file was not modified.")

if __name__ == "__main__":
    main()
