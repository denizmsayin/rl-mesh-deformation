import torch
from pytorch3d.loss import chamfer_distance

from rlmd.ops.sampling import sample_points_from_polylines


class ChamferMetric:
    """
    Chamfer distance between two batches of padded polylines.

    Each input is a (V, L, num_verts) tuple following the project's batched
    polygon convention:
        V         : (B, N_max, 2) float — padded vertex positions
        L         : (B, M_max, 2) long  — padded edge index pairs
        num_verts : (B,) long           — valid vertex/edge count per item

    Points are first sampled along each polyline (length-weighted) so both
    clouds have a fixed size of ``num_samples`` per item, then chamfer is
    computed in each direction separately and both are returned.
    """

    def __init__(self, num_samples: int = 1024, point_reduction: str = "mean", norm: int = 2):
        if point_reduction not in ("mean", "sum"):
            raise ValueError(f"point_reduction must be 'mean' or 'sum', got {point_reduction!r}")
        if norm not in (1, 2):
            raise ValueError(f"norm must be 1 or 2, got {norm!r}")

        self.num_samples = num_samples
        self.point_reduction = point_reduction
        self.norm = norm

    def _one_direction(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        d, _ = chamfer_distance(
            x=src,
            y=tgt,
            batch_reduction=None,
            point_reduction=self.point_reduction,
            norm=self.norm,
            single_directional=True,
        )
        return d

    def __call__(self, poly_a, poly_b):
        V_a, L_a, n_a = poly_a
        V_b, L_b, n_b = poly_b

        pts_a = sample_points_from_polylines(V_a, L_a, n_a, self.num_samples)
        pts_b = sample_points_from_polylines(V_b, L_b, n_b, self.num_samples)

        a_to_b = self._one_direction(pts_a, pts_b)
        b_to_a = self._one_direction(pts_b, pts_a)
        return a_to_b, b_to_a
