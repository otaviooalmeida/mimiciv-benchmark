"""Build disk-backed split indices and window-weighted normalization statistics."""
import argparse
from normalization import build_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional patient split seed; default remains unseeded, as before.')
    args = parser.parse_args()
    manifest = build_dataset(seed=args.seed)
    print('Dataset ready (train/validation/test):', manifest['lengths'])
