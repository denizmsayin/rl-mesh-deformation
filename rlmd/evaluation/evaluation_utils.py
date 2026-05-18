import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import packed_to_padded
import numpy as np
import csv







def make_packed_inputs(x: torch.Tensor, y: torch.Tensor):
    """
    given two tensors x and y with shapes:
        x: (num_points_1, D)
        y: (num_points_2, D)

    returns the inputs needed for evaluation and the original lengths:
        padded : (N_batches = 2, max_num_points, D)
        lengths : (original length x, original length y)
    could be useful somewhere.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must both have shape (num_points, D).")

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension D.")

    if x.device != y.device:
        raise ValueError("x and y must be on the same device.")

    # store lengths
    lengths = torch.tensor(
        [x.shape[0], y.shape[0]],
        dtype=torch.long,
        device=x.device
    )

    # concatenate into packed representation
    inputs = torch.cat([x, y], dim=0) # (F, D)

    first_idxs = torch.cat([
        torch.zeros(1, dtype=torch.long, device=x.device),
        torch.cumsum(lengths, dim=0)[:-1]
    ])

    # maximum number of points among the two tensors
    max_size = int(lengths.max().item())

    # create tensors adapt to evaluation by padding 
    padded = packed_to_padded(inputs, first_idxs, max_size)
    return padded, lengths



def chamfer(x: torch.Tensor, y: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor, weights: torch.Tensor,
            x_normals=None, y_normals=None, batch_reduction="mean", point_reduction="mean", norm=2,
            single_directional=False, abs_cosine=True):
    """
    inputs : 
            x : (N, P, D) clouds 1, torch.Tensor
            y : (N, P, D) clouds 2, torch.Tensor
    outputs : 
            chamf : chamfer distance, it's shape depends on the type of reductions
            chamf_normal : useful only if x-normals, y-normals are not None (not our case)  

    """

    N1, P1, D1 = x.shape
    N2, P2, D2 = y.shape

    if D1 != D2:
        raise RuntimeError(f"clouds x and y have points with different dimensions, lol wtf.")
    if N1 != N2:
        raise RuntimeError(f"clouds x and y have different number of batches dude.")

    chamf, chamf_normal = chamfer_distance(
        x=x,
        y=y,
        x_lengths=x_lengths,
        y_lengths=y_lengths,
        x_normals=x_normals,
        y_normals=y_normals,
        weights=weights,
        batch_reduction=batch_reduction,
        point_reduction=point_reduction,
        norm=norm,
        single_directional=single_directional,
        abs_cosine=abs_cosine
    )

    return chamf, chamf_normal

