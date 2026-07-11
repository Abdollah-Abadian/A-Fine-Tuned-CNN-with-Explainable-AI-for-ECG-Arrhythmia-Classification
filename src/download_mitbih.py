"""
Download the MIT-BIH Arrhythmia Database from PhysioNet.

Usage
-----
python src/download_mitbih.py --out data/raw
"""

import argparse
import os

import wfdb

ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def download(out_dir: str, records=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    records = records or ALL_RECORDS
    print(f"Downloading {len(records)} MIT-BIH records to {out_dir} ...")
    wfdb.dl_database("mitdb", dl_dir=out_dir, records=records)
    print("Download complete.")

    missing = [
        r for r in records
        if not os.path.exists(os.path.join(out_dir, f"{r}.dat"))
    ]
    if missing:
        raise RuntimeError(
            f"The following records failed to download: {missing}. "
            "Check network connectivity to physionet.org and re-run."
        )
    print("Verified all record triplets (.dat/.hea/.atr) are present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument(
        "--records", nargs="*", default=None,
        help="Optional subset of record IDs to download (default: all 48)",
    )
    args = parser.parse_args()
    download(args.out, args.records)
