#!/usr/bin/env python3
"""Remove direct identifiers from the raw oTree exports before deposit.

The original oTree "all_apps_wide" exports contain a payment-information page
(Stage_Payment_Info) on which subjects entered their name, school, student ID,
national ID number, address, residence-permit / passport numbers, postal code, nationality
and length of stay. These fields were collected only to pay subjects and are
never used in the analysis. This script drops them and leaves every other cell
byte-for-byte unchanged (values are handled as strings, never re-typed).

This script was run once by the authors on the confidential originals. It is
included for transparency; replicators do NOT need to run it, because the
de-identified files it produced are the ones shipped in build/input/.

Usage:
    python tools/deidentify_raw.py <original_input_dir> <deidentified_output_dir>
"""
import csv
import glob
import os
import re
import sys

# Fields on Stage_Payment_Info that identify a person.
PII_FIELDS = {
    "name", "school", "student_id", "id_number", "address", "address_code",
    "is_foreign", "arc", "passport", "nation", "stay",
}
PII_PATTERN = re.compile(r"^Stage_Payment_Info\.\d+\.player\.(" + "|".join(sorted(PII_FIELDS)) + r")$")


def deidentify(src, dst):
    with open(src, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    keep = [i for i, col in enumerate(header) if not PII_PATTERN.match(col)]
    dropped = [header[i] for i in range(len(header)) if i not in set(keep)]

    with open(dst, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow([header[i] for i in keep])
        for row in rows:
            writer.writerow([row[i] for i in keep])
    return dropped, len(rows)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src_dir, dst_dir = sys.argv[1], sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {src_dir}")
    for src in files:
        dst = os.path.join(dst_dir, os.path.basename(src))
        dropped, n = deidentify(src, dst)
        print(f"{os.path.basename(src)}: {n} rows, dropped {len(dropped)} identifier columns")


if __name__ == "__main__":
    main()
