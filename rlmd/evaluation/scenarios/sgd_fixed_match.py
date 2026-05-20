from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from rlmd.ops import (
    distance_loss,
    polyline_edge_loss,
    polyline_laplacian_smoothing,
    polyline_normal_consistency,
)


Polyline = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (V, L, num_verts)


def _inner_loss(V_src, deform, V_tgt, matchings, L_src, nv_src,
                w_data, w_edge, w_normal, w_laplacian, p):
    V = V_src + deform
    l_data = distance_loss(V, V_tgt, matchings, p=p)
    l_edge = polyline_edge_loss(V, L_src, nv_src)
    l_normal = polyline_normal_consistency(V, L_src, nv_src)
    l_laplacian = polyline_laplacian_smoothing(V, L_src, nv_src)
    total = (w_data * l_data
             + w_edge * l_edge
             + w_normal * l_normal
             + w_laplacian * l_laplacian)
    return total, V


_inner_loss_compiled = torch.compile(_inner_loss)


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

    def run(
        self,
        poly_src: Polyline,
        poly_tgt: Polyline,
        matcher,
        *,
        record_every: Optional[int] = None,
        record_max_batch: Optional[int] = None,
    ):
        V_src, L_src, nv_src = poly_src
        V_tgt, _, nv_tgt = poly_tgt

        with torch.no_grad():
            matchings = matcher(V_src, nv_src, V_tgt, nv_tgt)

        deform = torch.zeros_like(V_src, requires_grad=True)
        optimizer = torch.optim.SGD([deform], lr=self.lr, momentum=self.momentum)

        frames = [] if record_every is not None else None
        K = record_max_batch if record_max_batch is not None else V_src.shape[0]

        def _snapshot(V_now: torch.Tensor) -> None:
            frames.append(V_now[:K].detach().to("cpu", copy=True))

        for i in range(self.num_iters):
            optimizer.zero_grad()
            total, V = _inner_loss_compiled(
                V_src, deform, V_tgt, matchings, L_src, nv_src,
                self.w_data, self.w_edge, self.w_normal, self.w_laplacian,
                self.distance_p,
            )
            if frames is not None and i % record_every == 0:
                _snapshot(V)
            total.backward()
            optimizer.step()

        V_final = (V_src + deform).detach()
        if frames is not None:
            _snapshot(V_final)
            return V_final, torch.stack(frames, dim=0)
        return V_final
