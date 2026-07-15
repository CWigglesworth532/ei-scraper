#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


CURRENT_MATCH_FILE = "telos_matches_enriched_v3.csv"
PRIOR_REVIEW_FILE = "AZ supplier matches telos (1)(telos_exact_overlaps_deduped).csv"
OUTPUT_XLSX = "telos_client_review_merged.xlsx"


def read_prior_review(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="latin1",
    )


def normalize_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def add_client_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "client_confirmed_match",
        "client_confirmed_country",
        "client_confirmed_address",
        "client_confirmed_postcode",
        "client_notes",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = ""
    return df


def main():
    current = pd.read_csv(CURRENT_MATCH_FILE, dtype=str, keep_default_na=False)
    prior = read_prior_review(PRIOR_REVIEW_FILE)

    current = add_client_columns(current)

    # Join key: normalized supplier name
    current["_join_supplier_name"] = normalize_key(current["_supplier_match_name"])
    prior["_join_supplier_name"] = normalize_key(prior["supplier name"])

    # Tidy prior review columns
    prior = prior.rename(columns={
        "Country": "client_country_from_prior_review",
        "Country Match": "country_match_from_prior_review",
        "CY2025 Spend": "cy2025_spend",
        "original supplier name": "original_supplier_name_from_prior_review",
        "matched entity name": "matched_entity_name_from_prior_review",
        "country": "matched_country_from_prior_review",
        "tax_id": "tax_id_from_prior_review",
        "ei_register_name": "ei_register_name_from_prior_review",
        "source_url": "source_url_from_prior_review",
    })

    prior_keep = [
        "_join_supplier_name",
        "original_supplier_name_from_prior_review",
        "matched_entity_name_from_prior_review",
        "matched_country_from_prior_review",
        "tax_id_from_prior_review",
        "ei_register_name_from_prior_review",
        "source_url_from_prior_review",
        "client_country_from_prior_review",
        "country_match_from_prior_review",
        "cy2025_spend",
    ]
    prior_keep = [c for c in prior_keep if c in prior.columns]

    prior_small = prior[prior_keep].drop_duplicates(subset=["_join_supplier_name"], keep="first").copy()

    merged = current.merge(
        prior_small,
        how="left",
        on="_join_supplier_name",
    )

    # Normalize country-match values
    if "country_match_from_prior_review" in merged.columns:
        merged["country_match_from_prior_review"] = (
            merged["country_match_from_prior_review"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Suggested workflow flags
    merged["review_stage"] = ""
    merged.loc[
        merged["country_match_from_prior_review"] == "TRUE",
        "review_stage"
    ] = "ADDRESS_REVIEW"

    merged.loc[
        (merged["country_match_from_prior_review"] != "TRUE") &
        (merged["match_type"].fillna("").str.strip() != ""),
        "review_stage"
    ] = "COUNTRY_REVIEW"

    # Address completeness flags
    merged["has_address_data"] = merged["address"].fillna("").str.strip().ne("").map(lambda x: "YES" if x else "NO")
    merged["has_postcode_data"] = merged["postcode"].fillna("").str.strip().ne("").map(lambda x: "YES" if x else "NO")
    merged["has_city_data"] = merged["city"].fillna("").str.strip().ne("").map(lambda x: "YES" if x else "NO")

    # Prefill client country where already reviewed as true
    if "client_country_from_prior_review" in merged.columns:
        fill_mask = (
            merged["country_match_from_prior_review"] == "TRUE"
        ) & (
            merged["client_confirmed_country"].fillna("").str.strip() == ""
        )
        merged.loc[fill_mask, "client_confirmed_country"] = merged.loc[fill_mask, "client_country_from_prior_review"]

    # Create workbook tabs
    all_rows = merged.copy()

    exact_rows = merged[merged["match_type"] == "name_exact_norm"].copy()
    fuzzy_rows = merged[merged["match_type"] == "name_fuzzy"].copy()

    country_confirmed = merged[merged["country_match_from_prior_review"] == "TRUE"].copy()

    address_review = merged[
        (merged["country_match_from_prior_review"] == "TRUE") &
        (
            merged["address"].fillna("").str.strip().eq("") |
            merged["postcode"].fillna("").str.strip().eq("") |
            merged["city"].fillna("").str.strip().eq("")
        )
    ].copy()

    country_review = merged[
        merged["country_match_from_prior_review"] != "TRUE"
    ].copy()

    # Column order
    preferred = [
        "supplier_legal_name",
        "_supplier_match_name",
        "match_type",
        "match_score",
        "confidence",
        "matched_entity_name",
        "matched_register",
        "entity_name",
        "ei_registration_number",
        "country",
        "ccaa",
        "city",
        "postcode",
        "address",
        "tax_id",
        "client_country_from_prior_review",
        "country_match_from_prior_review",
        "cy2025_spend",
        "review_stage",
        "has_address_data",
        "has_postcode_data",
        "has_city_data",
        "client_confirmed_match",
        "client_confirmed_country",
        "client_confirmed_address",
        "client_confirmed_postcode",
        "client_notes",
    ]

    def reorder(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        return df[cols]

    all_rows = reorder(all_rows)
    exact_rows = reorder(exact_rows)
    fuzzy_rows = reorder(fuzzy_rows)
    country_confirmed = reorder(country_confirmed)
    address_review = reorder(address_review)
    country_review = reorder(country_review)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        all_rows.to_excel(writer, sheet_name="All_Matches", index=False)
        exact_rows.to_excel(writer, sheet_name="Exact", index=False)
        fuzzy_rows.to_excel(writer, sheet_name="Fuzzy", index=False)
        country_confirmed.to_excel(writer, sheet_name="Country_Confirmed", index=False)
        address_review.to_excel(writer, sheet_name="Address_Review", index=False)
        country_review.to_excel(writer, sheet_name="Country_Review", index=False)

        instructions = pd.DataFrame({
            "Instructions": [
                "client_supplier_name / supplier_legal_name is the original supplier name from the client source.",
                "country_match_from_prior_review = TRUE means the client already confirmed the country.",
                "Use Address_Review to focus only on rows where country is already confirmed but location fields still need checking.",
                "Use Country_Review for rows that still need country triangulation.",
                "Do not overwrite system-generated columns; fill only client_confirmed_* and client_notes if needed.",
            ]
        })
        instructions.to_excel(writer, sheet_name="Instructions", index=False)

    print(f"Saved: {OUTPUT_XLSX}")
    print(f"All rows: {len(all_rows)}")
    print(f"Exact rows: {len(exact_rows)}")
    print(f"Fuzzy rows: {len(fuzzy_rows)}")
    print(f"Country confirmed: {len(country_confirmed)}")
    print(f"Address review: {len(address_review)}")
    print(f"Country review: {len(country_review)}")


if __name__ == "__main__":
    main()
