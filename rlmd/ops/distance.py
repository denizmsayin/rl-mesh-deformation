from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points


@dataclass
class Matching:
    """
    A set of (src, tgt) index pairs between two padded point clouds.

    Fields:
        idx_src: (B, K) long — positions in P_src.
        idx_tgt: (B, K) long — positions in P_tgt.
        mask:    (B, K) bool — valid pairs (False for padding).
    """
    idx_src: torch.Tensor
    idx_tgt: torch.Tensor
    mask: torch.Tensor


def knn_match(P_src, n_src, P_tgt, n_tgt, bidirectional=True):
    """
    Nearest-neighbor matching between two padded point clouds.

    Args:
        P_src: (B, N_src, D) float.
        n_src: (B,) long — valid points per batch item in P_src.
        P_tgt, n_tgt: same for target.
        bidirectional: if True, returns [S->T, T->S]; else just [S->T].

    Returns:
        list of Matching objects.
    """
    B, N_src, _ = P_src.shape
    N_tgt = P_tgt.shape[1]
    device = P_src.device

    def _one_direction(A, n_A, B_pts, n_B, N_A):
        out = knn_points(A, B_pts, lengths1=n_A, lengths2=n_B, K=1)
        idx_A = torch.arange(N_A, device=device)[None, :].expand(B, -1)
        idx_B = out.idx.squeeze(-1)
        mask = torch.arange(N_A, device=device)[None, :] < n_A[:, None]
        return idx_A, idx_B, mask

    idx_src, idx_tgt, mask = _one_direction(P_src, n_src, P_tgt, n_tgt, N_src)
    matchings = [Matching(idx_src, idx_tgt, mask)]

    if bidirectional:
        idx_t, idx_s, mask_t = _one_direction(P_tgt, n_tgt, P_src, n_src, N_tgt)
        matchings.append(Matching(idx_src=idx_s, idx_tgt=idx_t, mask=mask_t))

    return matchings


def distance_loss(P_src, P_tgt, matchings, p=2):
    """
    Average L_p distance (to the p-th power) over matched pairs.

    For each Matching m, computes mean over valid pairs of
        sum_d |P_src[i,d] - P_tgt[j,d]|^p,
    averaged over batch. Returns the mean across matchings.

    p=2 gives squared-L2 (matches pytorch3d chamfer default). p=1 gives L1.

    Args:
        P_src: (B, N_src, D) float.
        P_tgt: (B, N_tgt, D) float.
        matchings: list of Matching.
        p: L_p exponent.

    Returns:
        scalar loss.
    """
    B = P_src.shape[0]
    batch = torch.arange(B, device=P_src.device)[:, None]

    losses = []
    for m in matchings:
        a = P_src[batch, m.idx_src]
        b = P_tgt[batch, m.idx_tgt]
        dist = (a - b).abs().pow(p).sum(dim=-1)
        mask = m.mask.to(dist.dtype)
        per_item = (dist * mask).sum(dim=-1) / mask.sum(dim=-1)
        losses.append(per_item.mean())

    return sum(losses) / len(losses)
