"""Batch-aligned CPU forecast shards for bounded-memory evaluation."""

import json
from pathlib import Path

import numpy as np

from evaluation import extract_targets, validate_forecasts


class ForecastShardStore:
    def __init__(self, directory, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale_shard in self.directory.glob("part-*.npz"):
            stale_shard.unlink()
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("format", "npz-shards-v1")
        self.metadata.setdefault("joint_arrays", ["generation", "samples_y", "samples_x", "info"])
        with (self.directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2, ensure_ascii=False)
        self.next_index = 0
        self.sample_offset = 0

    def add(self, generation, samples_y, samples_x, info):
        """Persist one aligned batch, assigning stable ordinals across shards."""
        generation = np.asarray(generation)
        samples_y = np.asarray(samples_y)
        samples_x = np.asarray(samples_x)
        info = np.asarray(info)
        targets = extract_targets(samples_y)
        validate_forecasts(generation, targets)
        batch_size = generation.shape[0]
        if samples_x.ndim != 3 or samples_x.shape[0] != batch_size:
            raise ValueError("samples_x must have one [4,Lx] row per generated sample")
        if info.ndim != 2 or info.shape[0] != batch_size:
            raise ValueError("info must have one metadata row per generated sample")
        start, stop = self.sample_offset, self.sample_offset + batch_size
        path = self.directory / f"part-{self.next_index:06d}.npz"
        np.savez_compressed(
            path,
            generation=generation,
            samples_y=samples_y,
            samples_x=samples_x,
            info=info,
            sample_ordinal=np.arange(start, stop, dtype=np.int64),
        )
        self.next_index += 1
        self.sample_offset = stop
        return path

    def shards(self):
        return sorted(self.directory.glob("part-*.npz"))

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as shard:
            return {name: shard[name] for name in shard.files}
