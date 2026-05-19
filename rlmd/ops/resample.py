import torch

from rlmd.models.polygon_cnn import check_sequential_l


def resample_uniform_polyline(V, L, num_verts, M, check_l=False):
    """
    Resample each closed polyline to M uniformly arc-length-spaced vertices.

    The first output vertex always coincides with V[..., 0, :] (s = 0), so the
    canonical orientation of the input polyline is preserved.

    Assumes L encodes the canonical sequential ordering
    ``L[b, i] == (i, (i+1) mod num_verts[b])`` for valid i. Set check_l=True to
    validate.

    Differentiable in V via gather + linear interpolation.

    Args:
        V: (B, N_max, D) float — padded vertex coords.
        L: (B, M_max, 2) long — edges; only consulted when check_l=True.
        num_verts: (B,) long — valid vertices per item.
        M: int — output vertex count.
        check_l: if True, validate sequential ordering.

    Returns:
        V_new:  (B, M, D) float — resampled vertices.
        L_new:  (B, M, 2) long — canonical sequential edges.
        nv_new: (B,) long — all equal to M.
    """
    if check_l:
        check_sequential_l(L, num_verts)

    B, N_max, D = V.shape
    device = V.device
    dtype = V.dtype

    # Index of the "next" vertex for each position, cyclic within each item's
    # valid run of length n. Padded positions don't matter; we set them to 0.
    idx = torch.arange(N_max, device=device)
    idx_b = idx[None, :].expand(B, -1)                           # (B, N_max)
    n = num_verts[:, None]                                       # (B, 1)
    valid = idx_b < n                                             # (B, N_max)
    n_safe = n.clamp(min=1)
    next_idx = torch.where(valid, (idx_b + 1) % n_safe, torch.zeros_like(idx_b))

    next_idx_d = next_idx.unsqueeze(-1).expand(-1, -1, D)        # (B, N_max, D)
    V_next = torch.gather(V, 1, next_idx_d)                      # (B, N_max, D)

    # Per-edge length, zeroed at padded positions.
    seg = torch.linalg.vector_norm(V_next - V, dim=-1)           # (B, N_max)
    seg = seg * valid.to(seg.dtype)

    # Cumulative arc length, padded with a leading zero so that
    # cum[b, i] = sum_{j<i} seg[b, j] and cum[b, n] = total perimeter.
    zero_col = torch.zeros(B, 1, device=device, dtype=seg.dtype)
    cum = torch.cat([zero_col, seg.cumsum(dim=1)], dim=1)        # (B, N_max + 1)

    P = cum.gather(1, n)                                         # (B, 1) total perimeter

    # Query arc-length positions s_k = k/M * P, k = 0..M-1 (k = M would duplicate s = 0).
    k = torch.arange(M, device=device, dtype=dtype)
    s = (k / M)[None, :] * P                                     # (B, M)

    # Find edge index e such that cum[e] <= s < cum[e+1]. searchsorted with
    # right=True returns the first index where cum > s, in [1, n]; subtracting 1
    # lands in [0, n-1].
    e = torch.searchsorted(cum, s, right=True) - 1               # (B, M)
    e = e.clamp(min=0, max=N_max - 1)

    e_d = e.unsqueeze(-1).expand(-1, -1, D)                      # (B, M, D)
    V_e = torch.gather(V, 1, e_d)                                # (B, M, D)
    V_e_next = torch.gather(V_next, 1, e_d)                      # (B, M, D)
    seg_e = seg.gather(1, e)                                     # (B, M)
    cum_e = cum.gather(1, e)                                     # (B, M)

    t = (s - cum_e) / seg_e.clamp(min=torch.finfo(seg.dtype).tiny)  # (B, M)
    V_new = V_e + t.unsqueeze(-1) * (V_e_next - V_e)             # (B, M, D)

    i_out = torch.arange(M, device=device)
    L_new = torch.stack([i_out, (i_out + 1) % M], dim=-1)
    L_new = L_new[None, :, :].expand(B, -1, -1).contiguous()
    nv_new = torch.full((B,), M, dtype=torch.long, device=device)

    return V_new, L_new, nv_new
