#!/usr/bin/env python3
"""Build the auditable, chunked MIMIC-IV Parquet event dataset."""

import argparse
from pathlib import Path

try:
    from .event_dataset import build_event_dataset
except ImportError:
    from event_dataset import build_event_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", default="MIMICIV", help="Directory containing icu/ and hosp/ CSV files")
    parser.add_argument("--output-dir", default="data", help="Authorized local output directory")
    parser.add_argument("--chunksize", type=int, default=250_000, help="CSV rows processed per chunk")
    parser.add_argument(
        "--max-partition-rows", type=int, default=2_000_000,
        help="Safety ceiling before one admission/subject partition is materialized",
    )
    parser.add_argument("--staging-buckets", type=int, default=64, help="Fixed staging bucket count")
    parser.add_argument(
        "--max-staging-bucket-rows", type=int, default=2_000_000,
        help="Safety ceiling before one staging bucket is materialized",
    )
    args = parser.parse_args()
    limits = (args.chunksize, args.max_partition_rows, args.staging_buckets,
              args.max_staging_bucket_rows)
    if min(limits) <= 0:
        parser.error("chunk and partition limits must be positive")
    report = build_event_dataset(
        Path(args.mimic_root), Path(args.output_dir), chunksize=args.chunksize,
        max_partition_rows=args.max_partition_rows, staging_buckets=args.staging_buckets,
        max_staging_bucket_rows=args.max_staging_bucket_rows,
    )
    print("Wrote {} events and {} minute aggregates".format(
        report["row_flow"]["final_events"],
        sum(row["aggregate_rows"] for row in report["row_flow"]["by_source"].values()),
    ))
    print("Cohort flow: {}/cohort_flow.json".format(args.output_dir))


if __name__ == "__main__":
    main()
