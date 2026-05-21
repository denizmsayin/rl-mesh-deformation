import torch


def polyline_edge_loss(V, L, num_verts, target_length=0.0):
    """
    Mean squared deviation of edge length from a target length.

    Sum (not mean) over batch is intentional — see the note in
    ``rlmd.ops.distance.distance_loss``.

    Args:
        V: (B, N_max, 2) float.
        L: (B, M_max, 2) long, padded with (0, 0).
        num_verts: (B,) long — valid edges per item (== valid verts for closed polylines).
        target_length: desired edge length. 0.0 shrinks edges.

    Returns:
        scalar loss, averaged per item then summed over batch.
    """
    B, M_max, _ = L.shape
    batch = torch.arange(B, device=V.device)[:, None]

    v0 = V[batch, L[..., 0]]
    v1 = V[batch, L[..., 1]]

    lengths = (v1 - v0).norm(dim=-1)
    mask = torch.arange(M_max, device=V.device)[None, :] < num_verts[:, None]

    loss = ((lengths - target_length) ** 2) * mask
    return (loss.sum(dim=-1) / num_verts).sum()


def polyline_laplacian_smoothing(V, L, num_verts):
    """
    Uniform-Laplacian smoothing loss. For each vertex v_i with neighbors S(i),
    computes ||v_i - mean(v_j for j in S(i))||, averages per item, then sums
    over batch. Matches pytorch3d's mesh_laplacian_smoothing(method='uniform')
    up to the batch reduction (we sum, they mean — see note in
    ``rlmd.ops.distance.distance_loss``).

    Args:
        V: (B, N_max, 2) float.
        L: (B, M_max, 2) long, padded with (0, 0).
        num_verts: (B,) long.

    Returns:
        scalar loss.
    """
    B, N_max, _ = V.shape
    M_max = L.shape[1]
    batch = torch.arange(B, device=V.device)[:, None]

    edge_mask = torch.arange(M_max, device=V.device)[None, :] < num_verts[:, None]
    emask = edge_mask.to(V.dtype).unsqueeze(-1)

    a = L[..., 0]
    b = L[..., 1]
    v0 = V[batch, a] * emask
    v1 = V[batch, b] * emask

    neighbor_sum = torch.zeros_like(V)
    neighbor_sum.index_put_((batch, a), v1, accumulate=True)
    neighbor_sum.index_put_((batch, b), v0, accumulate=True)

    degree = torch.zeros(B, N_max, device=V.device, dtype=V.dtype)
    eones = edge_mask.to(V.dtype)
    degree.index_put_((batch, a), eones, accumulate=True)
    degree.index_put_((batch, b), eones, accumulate=True)

    delta = V - neighbor_sum / degree.clamp(min=1.0).unsqueeze(-1)
    loss = delta.norm(dim=-1)

    return (loss.sum(dim=-1) / num_verts).sum()


def polyline_normal_consistency(V, L, num_verts):
    """
    Normal-consistency loss for closed 2D polylines. At each vertex, compares
    the incoming and outgoing edge directions via 1 - cos(d_in, d_out).
    Equivalent to comparing edge normals (rotating both by 90° preserves cos).
    Assumes L is consistently oriented.

    Sum (not mean) over batch — see note in ``rlmd.ops.distance.distance_loss``.

    Args:
        V: (B, N_max, 2) float.
        L: (B, M_max, 2) long, padded with (0, 0).
        num_verts: (B,) long.

    Returns:
        scalar loss.
    """
    B, N_max, _ = V.shape
    M_max = L.shape[1]
    batch = torch.arange(B, device=V.device)[:, None]

    edge_mask = torch.arange(M_max, device=V.device)[None, :] < num_verts[:, None]
    emask = edge_mask.to(V.dtype).unsqueeze(-1)

    a = L[..., 0]
    b = L[..., 1]
    d = (V[batch, b] - V[batch, a]) * emask

    d_out = torch.zeros_like(V)
    d_in = torch.zeros_like(V)
    d_out.index_put_((batch, a), d, accumulate=True)
    d_in.index_put_((batch, b), d, accumulate=True)

    cos = torch.cosine_similarity(d_in, d_out, dim=-1, eps=1e-8)
    vert_mask = torch.arange(N_max, device=V.device)[None, :] < num_verts[:, None]
    loss = (1 - cos) * vert_mask

    return (loss.sum(dim=-1) / num_verts).sum()
