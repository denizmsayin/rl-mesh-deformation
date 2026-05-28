"""Shared inner-loss for vertex-direct frozen-match SGD scenarios.

Both `SgdFixedMatchScenario` (match once) and `SgdRematchScenario` (re-match
every K iters) descend the same per-iteration objective: a vertex-direct data
term over the current matching plus edge / normal / laplacian regularizers.
Factored here so the two scenarios share one (compiled) implementation.
"""
import torch

from rlmd.ops import (
    distance_loss,
    polyline_edge_loss,
    polyline_laplacian_smoothing,
    polyline_normal_consistency,
)


def _inner_loss(V_src, deform, V_tgt, matchings, L_src, nv_src, ne_src,
                w_data, w_edge, w_normal, w_laplacian, p):
    V = V_src + deform
    l_data = distance_loss(V, V_tgt, matchings, p=p)
    l_edge = polyline_edge_loss(V, L_src, ne_src)
    l_normal = polyline_normal_consistency(V, L_src, nv_src, ne_src)
    l_laplacian = polyline_laplacian_smoothing(V, L_src, nv_src, ne_src)
    total = (w_data * l_data
             + w_edge * l_edge
             + w_normal * l_normal
             + w_laplacian * l_laplacian)
    return total, V


_inner_loss_compiled = torch.compile(_inner_loss)
