from dataclasses import dataclass
from typing import Tuple

import torch

from rlmd.ops import (
    distance_loss,
    polyline_edge_loss,
    polyline_laplacian_smoothing,
    polyline_normal_consistency,
)


Polyline = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (V, L, num_verts)


@dataclass
class SgdFixedMatchScenario:
    """
    SGD deformation with vertex-direct, frozen correspondences.

    Unlike `SgdScenario`, this scenario does NOT sample points from the
    polylines. The matcher is called once at t=0 directly on the polyline
    vertices, and the resulting (idx_src, idx_tgt) pairs are reused as the
    data term for every iteration:

        L_data = mean over matched pairs of ||V[i] - V_tgt[j]||^2

    Regularizers (edge / normal / laplacian) are applied to V_src + deform.

    Intended for learned-matcher training, where the inputs have already been
    uniformly resampled to a fixed-count polyline (see
    `rlmd.ops.resample_uniform_polyline`) so vertex indices map onto an
    arc-length grid.

    Gradient note: REINFORCE puts the policy gradient on log π(c), not on the
    matching indices themselves (which are discrete). The matcher call here
    runs under `torch.no_grad()` deliberately — for training the caller
    wraps a pre-sampled action in a `FixedMatcher` adapter and holds the
    `log_prob` separately; for eval the matcher is an argmax wrapper.
    """

    name: str = "sgd_fixed_match"
    num_iters: int = 200
    lr: float = 1.0
    momentum: float = 0.9
    w_data: float = 1.0
    w_edge: float = 1.0
    w_normal: float = 0.01
    w_laplacian: float = 0.1
    distance_p: int = 2

    def run(self, poly_src: Polyline, poly_tgt: Polyline, matcher) -> torch.Tensor:
        V_src, L_src, nv_src = poly_src
        V_tgt, _, nv_tgt = poly_tgt

        with torch.no_grad():
            matchings = matcher(V_src, nv_src, V_tgt, nv_tgt)

        deform = torch.zeros_like(V_src, requires_grad=True)
        optimizer = torch.optim.SGD([deform], lr=self.lr, momentum=self.momentum)

        for _ in range(self.num_iters):
            optimizer.zero_grad()
            V = V_src + deform
            l_data = distance_loss(V, V_tgt, matchings, p=self.distance_p)
            l_edge = polyline_edge_loss(V, L_src, nv_src)
            l_normal = polyline_normal_consistency(V, L_src, nv_src)
            l_laplacian = polyline_laplacian_smoothing(V, L_src, nv_src)
            total = (self.w_data * l_data
                     + self.w_edge * l_edge
                     + self.w_normal * l_normal
                     + self.w_laplacian * l_laplacian)
            total.backward()
            optimizer.step()

        return (V_src + deform).detach()
