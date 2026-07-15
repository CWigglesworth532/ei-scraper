#!/usr/bin/env python3
import csv
import sys

SNAP = sys.argv[1] if len(sys.argv) > 1 else "2026-02-18"

SRC = f"data/ie/charitiesregulator/register/register_{SNAP}.csv"
OUT = f"data/ie/charitiesregulator/register/register_{SNAP}.slim.csv"

FIELDS = [
    "Registered Charity Number",
    "Registered Charity Name",
    "Status",
    "Primary Address",
]

ENCODINGS = ["utf-8-sig", "cp1252", "iso-8859-1"]

def main():
    last = None

    for enc in ENCODINGS:
        try:
            rows = 0

            with open(SRC, "r", encoding=enc, newline="") as fin, \
                 open(OUT, "w", encoding="utf-8", newline="") as fout:

                # Skip first line (Effective Date line)
                fin.readline()

                reader = csv.DictReader(fin)

                missing = [f for f in FIELDS if f not in (reader.fieldnames or [])]
                if missing:
                    raise RuntimeError(f"Missing columns: {missing}. Found: {reader.fieldnames}")

                writer = csv.DictWriter(fout, fieldnames=FIELDS)
                writer.writeheader()

                for row in reader:
                    writer.writerow({k: (row.get(k, "") or "").strip() for k in FIELDS})
                    rows += 1

            print(f"OK encoding={enc} rows={rows}")
            return

        except Exception as e:
            last = e
            continue

    raise last


if __name__ == "__main__":
    main()
