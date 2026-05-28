from typing import Dict, Optional, Tuple

import torch
from pytorch3d.loss import chamfer_distance

from rlmd.ops.sampling import sample_points_from_polylines


class ChamferMetric:
    """
    Per-sample chamfer distances between two batches of padded polylines.

    Each input is a (V, L, num_verts) tuple following the project's batched
    polygon convention:
        V         : (B, N_max, 2) float — padded vertex positions
        L         : (B, M_max, 2) long  — padded edge index pairs
        num_verts : (B,) long           — valid vertex/edge count per item

    Points are first sampled along each polyline (length-weighted) so both
    clouds have a fixed size of ``num_samples`` per item, then chamfer is
    computed in each direction separately. The metric emits three sub-values
    per sample: a→b, b→a, and their mean.

    When ``with_normals=True``, outward normals are sampled alongside the
    points (cheap to compute on the same edges) and a normal-consistency
    term is reported in each direction, computed as ``1 - cos(n_src, n_tgt)``
    on chamfer-matched pairs. Set ``abs_cosine=True`` to use ``1 - |cos|``
    instead, which is direction-agnostic. Default is ``abs_cosine=False``
    because the project convention is CCW-oriented polylines with consistent
    outward normals, so a flipped normal is a real error worth surfacing.
    """

    name = "chamfer"

    def __init__(
        self,
        num_samples: int = 1024,
        point_reduction: str = "mean",
        norm: int = 2,
        with_normals: bool = False,
        abs_cosine: bool = False,
    ):
        if point_reduction not in ("mean", "sum"):
            raise ValueError(f"point_reduction must be 'mean' or 'sum', got {point_reduction!r}")
        if norm not in (1, 2):
            raise ValueError(f"norm must be 1 or 2, got {norm!r}")

        self.num_samples = num_samples
        self.point_reduction = point_reduction
        self.norm = norm
        self.with_normals = with_normals
        self.abs_cosine = abs_cosine

    def _one_direction(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_n: Optional[torch.Tensor] = None,
        tgt_n: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        d, dn = chamfer_distance(
            x=src,
            y=tgt,
            x_normals=src_n,
            y_normals=tgt_n,
            batch_reduction=None,
            point_reduction=self.point_reduction,
            norm=self.norm,
            single_directional=True,
            abs_cosine=self.abs_cosine,
        )
        return d, dn

    def __call__(self, poly_a, poly_b) -> Dict[str, torch.Tensor]:
        V_a, L_a, _, ne_a = poly_a
        V_b, L_b, _, ne_b = poly_b

        if self.with_normals:
            pts_a, nrm_a = sample_points_from_polylines(
                V_a, L_a, ne_a, self.num_samples, return_normals=True
            )
            pts_b, nrm_b = sample_points_from_polylines(
                V_b, L_b, ne_b, self.num_samples, return_normals=True
            )
        else:
            pts_a = sample_points_from_polylines(V_a, L_a, ne_a, self.num_samples)
            pts_b = sample_points_from_polylines(V_b, L_b, ne_b, self.num_samples)
            nrm_a = nrm_b = None

        d_a2b, dn_a2b = self._one_direction(pts_a, pts_b, nrm_a, nrm_b)
        d_b2a, dn_b2a = self._one_direction(pts_b, pts_a, nrm_b, nrm_a)

        out = {
            "chamfer_a2b": d_a2b,
            "chamfer_b2a": d_b2a,
            "chamfer_sym": 0.5 * (d_a2b + d_b2a),
        }
        if self.with_normals:
            out["normal_a2b"] = dn_a2b
            out["normal_b2a"] = dn_b2a
            out["normal_sym"] = 0.5 * (dn_a2b + dn_b2a)
        return out
