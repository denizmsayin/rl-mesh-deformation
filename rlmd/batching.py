import torch


def pad_polylines(V_list, L_list, device=None, dtype=torch.float32):
    """
    Collate a list of per-shape polylines into a padded batch.

    Args:
        V_list: list of (N_i, 2) numpy arrays.
        L_list: list of (M_i, 2) numpy arrays. For a closed polyline M_i == N_i;
            for a welded grid M_i and N_i differ.
        device, dtype: target device and float dtype for V. L is always long.

    Returns:
        V: (B, N_max, 2) float tensor.
        L: (B, M_max, 2) long tensor, padded with -1 (sentinel for unused rows).
        num_verts: (B,) long tensor — valid vertices per item.
        num_edges: (B,) long tensor — valid edges per item.
    """
    B = len(V_list)
    N_max = max(v.shape[0] for v in V_list)
    M_max = max(l.shape[0] for l in L_list)

    V = torch.zeros(B, N_max, 2, dtype=dtype, device=device)
    L = torch.full((B, M_max, 2), -1, dtype=torch.long, device=device)
    num_verts = torch.tensor([v.shape[0] for v in V_list], dtype=torch.long, device=device)
    num_edges = torch.tensor([l.shape[0] for l in L_list], dtype=torch.long, device=device)

    for i, (v, l) in enumerate(zip(V_list, L_list)):
        V[i, :v.shape[0]] = torch.as_tensor(v, dtype=dtype)
        L[i, :l.shape[0]] = torch.as_tensor(l, dtype=torch.long)

    return V, L, num_verts, num_edges
