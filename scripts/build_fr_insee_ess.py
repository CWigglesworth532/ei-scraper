import os
import pandas as pd


# Input: full SIRENE "Unités légales" file
IN_PATH = "data/fr/insee/raw/StockUniteLegale_utf8.csv"

# Output: filtered ESS-only file
OUT_PATH = "data/fr/insee/fr_insee_ess_unites_legales.csv"


# Try comma first. If it fails, change to ";"
SEP = ","

# How many rows to read at once (memory-safe)
CHUNKSIZE = 300_000


# Columns we want to keep (if present)
WANTED_COLUMNS = [
    "siren",
    "denominationUniteLegale",
    "denominationUsuelle1UniteLegale",
    "categorieJuridiqueUniteLegale",
    "economieSocialeSolidaireUniteLegale",
    "nicSiegeUniteLegale",
    "codePostalSiegeUniteLegale",
    "libelleCommuneSiegeUniteLegale",
]


def main():

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("Reading header...")

    # Read header only
    head = pd.read_csv(
        IN_PATH,
        sep=SEP,
        dtype=str,
        nrows=1
    )

    if "economieSocialeSolidaireUniteLegale" not in head.columns:
        raise ValueError(
            "Missing economieSocialeSolidaireUniteLegale column.\n"
            "Try changing SEP to ';' if this fails."
        )

    # Keep only columns that actually exist
    usecols = [c for c in WANTED_COLUMNS if c in head.columns]

    print("Using columns:")
    for c in usecols:
        print(" -", c)

    first_write = True
    total_rows = 0

    print("Processing file in chunks...")

    for chunk in pd.read_csv(
        IN_PATH,
        sep=SEP,
        dtype=str,
        usecols=usecols,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):

        # Filter ESS = "O" (Oui)
        ess = chunk[
            chunk["economieSocialeSolidaireUniteLegale"]
            .astype(str)
            .str.upper()
            .eq("O")
        ]

        if ess.empty:
            continue

        ess.to_csv(
            OUT_PATH,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
            encoding="utf-8",
        )

        first_write = False
        total_rows += len(ess)

        print("Written so far:", total_rows)

    print("\nDONE")
    print("Output file:", OUT_PATH)
    print("Total ESS rows:", total_rows)


if __name__ == "__main__":
    main()
