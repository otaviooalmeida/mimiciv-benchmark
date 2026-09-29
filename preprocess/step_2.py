#!/usr/bin/env python3
"""Stream step-1 events into complete per-stay Parquet shards."""

import argparse

try:
    from .streaming_step2 import build_legacy_step2
except ImportError:  # executed as a script from preprocess/
    from streaming_step2 import build_legacy_step2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--missing-availability-policy", choices=("exclude", "measurement_time"),
        default="exclude",
        help="Explicit sensitivity fallback for missing storetime; default excludes those events.",
    )
    parser.add_argument("--data-dir", default="data", help="Preprocessing data directory")
    parser.add_argument("--chunksize", type=int, default=250_000, help="CSV rows held per chunk")
    parser.add_argument(
        "--max-stay-rows", type=int, default=2_000_000,
        help="Safety ceiling for any one stay before exact aggregation",
    )
    parser.add_argument(
        "--staging-buckets", type=int, default=64,
        help="Stable hash buckets used to bound staging file count",
    )
    parser.add_argument(
        "--max-staging-bucket-rows", type=int, default=2_000_000,
        help="Safety ceiling before materializing one staging bucket",
    )
    args = parser.parse_args()
    if min(args.chunksize, args.max_stay_rows, args.staging_buckets, args.max_staging_bucket_rows) <= 0:
        parser.error("chunk and partition limits must be positive")
    report = build_legacy_step2(
        args.data_dir, chunksize=args.chunksize,
        missing_availability_policy=args.missing_availability_policy,
        max_stay_rows=args.max_stay_rows, staging_buckets=args.staging_buckets,
        max_staging_bucket_rows=args.max_staging_bucket_rows,
    )
    print("Wrote {} stay partitions ({} aggregate rows)".format(
        report["stay_partition_count"], report["final_rows"],
    ))
    print("Availability audit: {}/availability_audit.json".format(args.data_dir))


if __name__ == "__main__":
    main()
