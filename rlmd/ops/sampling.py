import torch


def sample_points_from_polylines(V, L, num_verts, num_samples, return_normals=False):
    """
    Length-weighted uniform sampling of points from a batch of padded closed polylines.

    Args:
        V: (B, N_max, 2) float — padded vertex positions.
        L: (B, M_max, 2) long — padded edges as index pairs. For closed polylines
            M_max == N_max; pad rows with (0, 0) so they are degenerate self-loops
            with zero length.
        num_verts: (B,) long — number of valid vertices (== valid edges) per item.
        num_samples: int — points to draw per item.
        return_normals: if True, also return per-sample outward unit normals.
            Assumes input polylines are oriented counter-clockwise (interior on the
            left of each edge direction), which is the project convention — base
            shapes in rlmd.data.generation are CCW and transforms have positive
            determinant.

    Returns:
        points: (B, num_samples, 2) float — samples, differentiable w.r.t. V.
        normals: (B, num_samples, 2) float — only if return_normals=True. For an
            edge with direction d = v1 - v0, normal = (d_y, -d_x) / ‖d‖.
    """
    B, M_max, _ = L.shape
    batch = torch.arange(B, device=V.device)[:, None]

    v0 = V[batch, L[..., 0]]
    v1 = V[batch, L[..., 1]]

    lengths = (v1 - v0).norm(dim=-1)
    mask = torch.arange(M_max, device=V.device)[None, :] < num_verts[:, None]
    weights = lengths * mask

    edge_idx = torch.multinomial(weights, num_samples, replacement=True)
    a = v0[batch, edge_idx]
    b = v1[batch, edge_idx]

    t = torch.rand(B, num_samples, 1, device=V.device, dtype=V.dtype)
    points = a + t * (b - a)

    if not return_normals:
        return points

    d = b - a
    n = torch.stack((d[..., 1], -d[..., 0]), dim=-1)
    normals = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return points, normals
