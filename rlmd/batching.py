import torch


def pad_polylines(V_list, L_list, device=None, dtype=torch.float32):
    """
    Collate a list of per-shape polylines into a padded batch.

    Args:
        V_list: list of (N_i, 2) numpy arrays.
        L_list: list of (M_i, 2) numpy arrays. For closed polylines M_i == N_i.
        device, dtype: target device and float dtype for V. L is always long.

    Returns:
        V: (B, N_max, 2) float tensor.
        L: (B, M_max, 2) long tensor, padded with (0, 0).
        num_verts: (B,) long tensor.
    """
    B = len(V_list)
    N_max = max(v.shape[0] for v in V_list)
    M_max = max(l.shape[0] for l in L_list)

    V = torch.zeros(B, N_max, 2, dtype=dtype, device=device)
    L = torch.zeros(B, M_max, 2, dtype=torch.long, device=device)
    num_verts = torch.tensor([v.shape[0] for v in V_list], dtype=torch.long, device=device)

    for i, (v, l) in enumerate(zip(V_list, L_list)):
        V[i, :v.shape[0]] = torch.as_tensor(v, dtype=dtype)
        L[i, :l.shape[0]] = torch.as_tensor(l, dtype=torch.long)

    return V, L, num_verts
