#!/usr/bin/env python3
"""Create patient splits and online-normalized dataset shards."""

import argparse

try:
    from .streaming_step4 import build_legacy_step4
except ImportError:  # executed as a script from preprocess/
    from streaming_step4 import build_legacy_step4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Seed for patient-level splits")
    parser.add_argument("--data-dir", default="data", help="Preprocessing data directory")
    args = parser.parse_args()
    if args.seed is None:
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            args.seed = yaml.safe_load(file).get("seed", 2026)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    report = build_legacy_step4(args.data_dir, seed=args.seed)
    print("Wrote sharded normalized dataset: {}".format(report["split_counts"]))
    print("Dataset manifest: {}/dataset.pkl".format(args.data_dir))


if __name__ == "__main__":
    main()
