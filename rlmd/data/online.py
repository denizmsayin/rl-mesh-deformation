import random
from dataclasses import dataclass
from typing import List

import torch

from rlmd.data.generation import ShapeGenerator


@dataclass
class OnlineShapeSampler:
    """
    Streams batches of transformed shapes on-device by picking one shape spec
    per call and running it through `ShapeGenerator.generate_batch_torch`.

    Designed for training, where infinite fresh samples are preferable to a
    fixed on-disk corpus. Each batch is homogeneous (one shape type) and the
    shape type is drawn uniformly at random across calls.

    Args:
        shape_specs: list of dicts. Each must have `shape` (one of the
            generator's registered keys) and optionally `num_points`, plus any
            shape-class kwargs (e.g. `n_tips` for Star) and an optional `name`
            for logging.
        transform: dict with `translation_range`, `scale_range`,
            `rotation_range` (each a 2-tuple), and optional `isotropic_scale`.
        seed: master seed; the shape-choice RNG is seeded here, and each
            generated batch uses a fresh derived seed so successive calls
            produce different transforms.
    """
    shape_specs: List[dict]
    transform: dict
    seed: int = 0

    def __post_init__(self):
        self._gen = ShapeGenerator()
        self._rng = random.Random(int(self.seed))
        self._batch_counter = 0

    def next_batch(self, batch_size, device, dtype=torch.float32):
        spec = dict(self._rng.choice(self.shape_specs))
        name = spec.get("name", spec["shape"])

        self._batch_counter += 1
        batch_seed = int(self.seed) * 1_000_003 + self._batch_counter

        batch = self._gen.generate_batch_torch(
            instance_name=name,
            shape_spec=spec,
            transform_cfg=self.transform,
            batch_size=int(batch_size),
            seed=batch_seed,
            device=device,
            dtype=dtype,
        )

        V = batch.points()                                              # (B, P, 2)
        P = V.shape[1]
        L = batch.edges[None].expand(int(batch_size), -1, -1).contiguous()
        num_verts = torch.full((int(batch_size),), P, dtype=torch.long, device=device)
        shape_names = [name] * int(batch_size)
        return V, L, num_verts, shape_names
