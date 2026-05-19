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

    A single `torch.Generator` drives both the shape-spec choice and the
    per-batch integer seed handed to the backend, so reproducibility is just
    "set `seed` to the same value".

    Args:
        shape_specs: list of dicts. Each must have `shape` (one of the
            generator's registered keys) and optionally `num_points`, plus any
            shape-class kwargs (e.g. `n_tips` for Star) and an optional `name`
            for logging.
        transform: dict with `translation_range`, `scale_range`,
            `rotation_range` (each a 2-tuple), and optional `isotropic_scale`.
        seed: master seed for the internal torch.Generator.
    """
    shape_specs: List[dict]
    transform: dict
    seed: int = 0

    def __post_init__(self):
        self._gen = ShapeGenerator()
        # CPU generator: only used for drawing small Python ints (spec choice
        # and the int seed forwarded to generate_batch_torch). Device-side
        # randomness for the actual transforms is owned by the backend.
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.seed))

    def next_batch(self, batch_size, device, dtype=torch.float32):
        spec_idx = int(torch.randint(0, len(self.shape_specs), (), generator=self._rng).item())
        spec = dict(self.shape_specs[spec_idx])
        name = spec.get("name", spec["shape"])
        batch_seed = int(torch.randint(0, 2**31 - 1, (), generator=self._rng).item())

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
