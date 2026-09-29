#!/usr/bin/env python3
"""Create at most one causally available forecast window per complete ICU stay."""

import argparse

try:
    from .streaming_step3 import build_legacy_step3
except ImportError:  # executed as a script from preprocess/
    from streaming_step3 import build_legacy_step3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Preprocessing data directory")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Concurrent complete-stay workers; each worker reads one Parquet stay file",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    report = build_legacy_step3(args.data_dir, workers=args.workers)
    print("Processed {} stays; produced {} samples with {} worker(s)".format(
        report["stay_files_processed"], report["sampled_stays"], report["workers"],
    ))


if __name__ == "__main__":
    main()
