from typing import Dict

import torch


class SegmentStdMetric:
    """
    Per-sample standard deviation of edge lengths within a polyline.

    Operates on ``poly_pred`` only — measures structural uniformity of the
    deformed mesh. ``poly_tgt`` is accepted to satisfy the Metric protocol but
    is unused. Each input is a (V, L, num_verts) tuple following the project's
    batched polygon convention:
        V         : (B, N_max, D) float — padded vertex positions
        L         : (B, M_max, 2) long  — padded edge index pairs (pad = -1)
        num_verts : (B,) long           — valid vertex count per item

    Edges whose endpoints fall outside ``[0, num_verts[b])`` are masked out
    before the std is computed. Samples with no valid edges receive a value
    of 0.
    """

    name = "segment_std"

    def __init__(self, unbiased: bool = False):
        self.unbiased = unbiased

    def __call__(self, poly_pred, poly_tgt) -> Dict[str, torch.Tensor]:
        del poly_tgt
        V, L, n = poly_pred
        B = V.shape[0]
        M = L.shape[1]

        n_b = n.view(B, 1, 1)
        valid = ((L >= 0) & (L < n_b)).all(dim=-1)  # (B, M)

        batch_idx = torch.arange(B, device=V.device).view(B, 1, 1).expand(B, M, 2)
        endpoints = V[batch_idx, L]  # (B, M, 2, D); pad rows masked out below

        diff = endpoints[..., 0, :] - endpoints[..., 1, :]  # (B, M, D)
        seg_lens = torch.linalg.vector_norm(diff, dim=-1)   # (B, M)

        mask = valid.to(seg_lens.dtype)
        counts = mask.sum(dim=-1)  # (B,)

        safe_counts = counts.clamp(min=1)
        means = (seg_lens * mask).sum(dim=-1) / safe_counts
        sq = (seg_lens - means.unsqueeze(-1)) ** 2 * mask

        if self.unbiased:
            denom = (counts - 1).clamp(min=1)
            var = sq.sum(dim=-1) / denom
            var = torch.where(counts >= 2, var, torch.zeros_like(var))
        else:
            var = sq.sum(dim=-1) / safe_counts
            var = torch.where(counts >= 1, var, torch.zeros_like(var))

        std = torch.sqrt(var)
        return {"segment_std": std}
