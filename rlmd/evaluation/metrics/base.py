from typing import Protocol, Dict

import torch


class Metric(Protocol):
    name: str

    def __call__(
        self, poly_pred, poly_tgt
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            poly_pred, poly_tgt: (V, L, num_verts) tuples (padded).

        Returns:
            dict mapping a sub-metric name to a (B,) tensor of per-sample values.
        """
        ...
