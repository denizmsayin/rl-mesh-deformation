import numpy as np
import torch

from rlmd.batching import pad_polylines
from rlmd.evaluation.matchers.knn_3d import Knn3dMatcher
from rlmd.evaluation.scenarios import SgdFixedMatchScenario
from rlmd.ops import resample_uniform_polyline


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int64)


def _circle(n, rx=1.0, ry=1.0, phase=0.0, cx=0.0, cy=0.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    x = cx + rx * np.cos(angles)
    y = cy + ry * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float32)


def _mean_pair_distance(V_src, V_tgt, matchings):
    """Average over matched pairs of ||V_src[i] - V_tgt[j]||^2 ."""
    losses = []
    B = V_src.shape[0]
    batch = torch.arange(B)[:, None]
    for m in matchings:
        a = V_src[batch, m.idx_src]
        b = V_tgt[batch, m.idx_tgt]
        d = ((a - b) ** 2).sum(dim=-1)
        mask = m.mask.to(d.dtype)
        losses.append((d * mask).sum(dim=-1) / mask.sum(dim=-1))
    return torch.stack(losses, dim=0).mean(dim=0)  # (B,)


def test_fixed_match_runs_and_reduces_data_term():
    # Source: unit circle. Target: ellipse 2x wide, 0.5x tall, shifted.
    # After resampling both to M=64 and using NN matching at t=0, frozen-match
    # SGD should reduce the data term substantially.
    V_src_np = _circle(20, rx=1.0, ry=1.0)
    V_tgt_np = _circle(35, rx=2.0, ry=0.5, cx=0.3, cy=-0.2)

    V_src, L_src, nv_src = pad_polylines([V_src_np], [_sequential_l(20)])
    V_tgt, L_tgt, nv_tgt = pad_polylines([V_tgt_np], [_sequential_l(35)])

    M = 64
    V_src_r, L_src_r, nv_src_r = resample_uniform_polyline(V_src, L_src, nv_src, M)
    V_tgt_r, L_tgt_r, nv_tgt_r = resample_uniform_polyline(V_tgt, L_tgt, nv_tgt, M)

    matcher = Knn3dMatcher(bidirectional=False)

    # Initial data term using the same matcher.
    with torch.no_grad():
        m0 = matcher(V_src_r, nv_src_r, V_tgt_r, nv_tgt_r)
        init_loss = _mean_pair_distance(V_src_r, V_tgt_r, m0).item()

    scenario = SgdFixedMatchScenario(
        num_iters=200,
        lr=1e-1,
        momentum=0.9,
        w_data=1.0,
        w_edge=1.0,
        w_normal=0.01,
        w_laplacian=0.1,
    )
    V_final = scenario.run(
        (V_src_r, L_src_r, nv_src_r),
        (V_tgt_r, L_tgt_r, nv_tgt_r),
        matcher,
    )

    assert V_final.shape == V_src_r.shape
    assert torch.isfinite(V_final).all()

    with torch.no_grad():
        # Same frozen matches as scenario uses; recompute to evaluate.
        m = matcher(V_src_r, nv_src_r, V_tgt_r, nv_tgt_r)
        final_loss = _mean_pair_distance(V_final, V_tgt_r, m).item()

    assert final_loss < 0.25 * init_loss, (init_loss, final_loss)


def test_fixed_match_runs_batched():
    # Two pairs in a batch with different shapes.
    V_src_list = [_circle(20, rx=1.0, ry=1.0), _circle(25, rx=0.8, ry=0.8, phase=0.3)]
    V_tgt_list = [_circle(30, rx=1.5, ry=0.6), _circle(40, rx=1.2, ry=1.2, cy=0.4)]

    V_src, L_src, nv_src = pad_polylines(
        V_src_list, [_sequential_l(v.shape[0]) for v in V_src_list])
    V_tgt, L_tgt, nv_tgt = pad_polylines(
        V_tgt_list, [_sequential_l(v.shape[0]) for v in V_tgt_list])

    M = 48
    V_src_r, L_src_r, nv_src_r = resample_uniform_polyline(V_src, L_src, nv_src, M)
    V_tgt_r, L_tgt_r, nv_tgt_r = resample_uniform_polyline(V_tgt, L_tgt, nv_tgt, M)

    scenario = SgdFixedMatchScenario(num_iters=50)
    V_final = scenario.run(
        (V_src_r, L_src_r, nv_src_r),
        (V_tgt_r, L_tgt_r, nv_tgt_r),
        Knn3dMatcher(bidirectional=False),
    )

    assert V_final.shape == (2, M, 2)
    assert torch.isfinite(V_final).all()
