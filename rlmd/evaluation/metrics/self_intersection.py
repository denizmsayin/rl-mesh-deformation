from typing import Dict

import torch


class SelfIntersectionMetric:
    """
    Per-sample count of self-intersections in a 2D polyline.

    Operates on ``poly_pred`` only — measures how badly the deformed mesh
    crosses itself. ``poly_tgt`` is accepted to satisfy the Metric protocol
    but is unused. Each input is a (V, L, num_verts) tuple following the
    project's batched polygon convention:
        V         : (B, N_max, 2) float — padded vertex positions
        L         : (B, M_max, 2) long  — padded edge index pairs (pad = -1)
        num_verts : (B,) long           — valid vertex/edge count per item

    Two edges count as intersecting when their open interiors strictly cross
    (proper intersection). Edges that share a vertex are excluded — they are
    adjacent in the polyline and touch at the shared endpoint, which is not
    a self-intersection. Collinear-overlap and T-junction cases yield a zero
    orientation and are not counted; this is a stop-gap, not an exact count
    over all geometric configurations.

    Each crossing pair is counted exactly once.
    """

    name = "self_intersection"

    def __call__(self, poly_pred, poly_tgt) -> Dict[str, torch.Tensor]:
        del poly_tgt
        V, L, n, _ = poly_pred
        B, M, _ = L.shape

        n_b = n.view(B, 1, 1)
        valid = ((L >= 0) & (L < n_b)).all(dim=-1)  # (B, M)

        L_safe = L.clamp(min=0)
        batch_idx = torch.arange(B, device=V.device).view(B, 1, 1).expand(B, M, 2)
        endpoints = V[batch_idx, L_safe]  # (B, M, 2, 2)
        A = endpoints[..., 0, :]  # (B, M, 2)
        Bp = endpoints[..., 1, :]  # (B, M, 2)

        A_i = A.unsqueeze(2)   # (B, M, 1, 2)
        B_i = Bp.unsqueeze(2)
        A_j = A.unsqueeze(1)   # (B, 1, M, 2)
        B_j = Bp.unsqueeze(1)

        def _cross(p, q, r):
            return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - \
                   (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

        o1 = _cross(A_i, B_i, A_j)
        o2 = _cross(A_i, B_i, B_j)
        o3 = _cross(A_j, B_j, A_i)
        o4 = _cross(A_j, B_j, B_i)

        proper = (torch.sign(o1) * torch.sign(o2) < 0) & \
                 (torch.sign(o3) * torch.sign(o4) < 0)

        i_idx = torch.arange(M, device=V.device)
        upper = i_idx.view(M, 1) < i_idx.view(1, M)  # (M, M)

        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, M, M)

        Li0 = L_safe[..., 0].unsqueeze(2)  # (B, M, 1)
        Li1 = L_safe[..., 1].unsqueeze(2)
        Lj0 = L_safe[..., 0].unsqueeze(1)  # (B, 1, M)
        Lj1 = L_safe[..., 1].unsqueeze(1)
        no_share = (Li0 != Lj0) & (Li0 != Lj1) & (Li1 != Lj0) & (Li1 != Lj1)

        keep = upper & valid_pair & no_share
        count = (proper & keep).sum(dim=(1, 2)).to(V.dtype)
        return {"self_intersection": count}
