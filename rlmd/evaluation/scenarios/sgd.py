from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from rlmd.ops import (
    distance_loss,
    polyline_edge_loss,
    polyline_laplacian_smoothing,
    polyline_normal_consistency,
    sample_points_from_polylines,
)


Polyline = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # (V, L, num_verts, num_edges)


@dataclass
class SgdScenario:
    """
    Iterative SGD deformation process.

    Optimizes an additive deform on V_src so that the resulting polyline matches
    V_tgt under a chamfer-style data term (computed via the given Matcher) plus
    edge / normal / laplacian regularizers. Returns the final V.
    """

    name: str = "sgd_default"
    num_iters: int = 2000
    num_samples: int = 500
    lr: float = 1.0
    momentum: float = 0.9
    w_chamfer: float = 1.0
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
        V_src, L_src, nv_src, ne_src = poly_src
        V_tgt, L_tgt, nv_tgt, ne_tgt = poly_tgt

        deform = torch.zeros_like(V_src, requires_grad=True)
        optimizer = torch.optim.SGD([deform], lr=self.lr, momentum=self.momentum)

        n_samples_src = torch.full((V_src.shape[0],), self.num_samples,
                                   dtype=torch.long, device=V_src.device)
        n_samples_tgt = torch.full((V_tgt.shape[0],), self.num_samples,
                                   dtype=torch.long, device=V_tgt.device)

        frames = [] if record_every is not None else None
        match_frames = [] if record_every is not None else None
        K = record_max_batch if record_max_batch is not None else V_src.shape[0]

        def _snapshot(V_now: torch.Tensor) -> None:
            V_snap = V_now[:K].detach().to("cpu", copy=True)
            frames.append(V_snap)
            if match_frames is not None:
                with torch.no_grad():
                    ms = matcher(
                        (V_now[:K], L_src[:K], nv_src[:K], ne_src[:K]),
                        (V_tgt[:K], L_tgt[:K], nv_tgt[:K], ne_tgt[:K]),
                    )
                match_frames.append(ms[0].idx_tgt.detach().cpu())

        for i in range(self.num_iters):
            optimizer.zero_grad()
            V = V_src + deform
            if frames is not None and i % record_every == 0:
                _snapshot(V)
            P = sample_points_from_polylines(V, L_src, ne_src, self.num_samples)
            P_tgt = sample_points_from_polylines(V_tgt, L_tgt, ne_tgt, self.num_samples)
            matchings = matcher((P, None, n_samples_src, None),
                                (P_tgt, None, n_samples_tgt, None))
            l_chamfer = distance_loss(P, P_tgt, matchings, p=self.distance_p)
            l_edge = polyline_edge_loss(V, L_src, ne_src)
            l_normal = polyline_normal_consistency(V, L_src, nv_src, ne_src)
            l_laplacian = polyline_laplacian_smoothing(V, L_src, nv_src, ne_src)
            total = (self.w_chamfer * l_chamfer
                     + self.w_edge * l_edge
                     + self.w_normal * l_normal
                     + self.w_laplacian * l_laplacian)
            total.backward()
            optimizer.step()

        V_final = (V_src + deform).detach()
        if frames is not None:
            _snapshot(V_final)
            return V_final, torch.stack(frames, dim=0), torch.stack(match_frames, dim=0)
        return V_final
