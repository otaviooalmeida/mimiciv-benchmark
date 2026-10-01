#!/usr/bin/env python3
"""Stream mapped MIMIC-IV events into the legacy step-2 input contract."""

import argparse

try:
    from .streaming_step1 import (
        DEFAULT_CHUNKSIZE,
        MAX_CHUNKSIZE,
        build_legacy_step1,
    )
except ImportError:  # executed as a script from preprocess/
    from streaming_step1 import DEFAULT_CHUNKSIZE, MAX_CHUNKSIZE, build_legacy_step1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", default="MIMICIV", help="Directory containing icu/ and hosp/ CSV files")
    parser.add_argument("--output-dir", default="data", help="Preprocessing output directory")
    parser.add_argument(
        "--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
        help="Source rows held per table chunk (default: 1000000; max: 2000000)",
    )
    args = parser.parse_args()
    if not 0 < args.chunksize <= MAX_CHUNKSIZE:
        parser.error("--chunksize must be between 1 and {}".format(MAX_CHUNKSIZE))
    report = build_legacy_step1(args.mimic_root, args.output_dir, chunksize=args.chunksize)
    print("Wrote {} mapped legacy events using chunks of {} rows".format(
        report["rows"].get("rows_written", 0), args.chunksize,
    ))
    print("Step-1 flow report: {}/step_1_flow.json".format(args.output_dir))


if __name__ == "__main__":
    main()
