import numpy as np
import torch

from rlmd.batching import pad_polylines
from rlmd.ops import resample_uniform_polyline


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int64)


def _circle(n, radius=1.0, phase=0.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)


def _square(side=1.0, per_edge=8):
    """CCW from (-s/2, -s/2). 4 * per_edge vertices, no duplicate corners."""
    s = side
    half = s / 2.0
    t = np.linspace(0.0, 1.0, per_edge, endpoint=False, dtype=np.float32)
    edges = [
        np.stack([-half + t * s, np.full_like(t, -half)], axis=1),
        np.stack([np.full_like(t, half), -half + t * s], axis=1),
        np.stack([half - t * s, np.full_like(t, half)], axis=1),
        np.stack([np.full_like(t, -half), half - t * s], axis=1),
    ]
    return np.concatenate(edges, axis=0)


def _perimeter(V):
    diffs = np.roll(V, -1, axis=0) - V
    return float(np.linalg.norm(diffs, axis=1).sum())


def _segment_lengths(V):
    diffs = np.roll(V, -1, axis=0) - V
    return np.linalg.norm(diffs, axis=1)


def _arc_length_along(V_input, pts):
    """For each point in `pts`, return its arc-length position along the closed
    polyline V_input. Assumes points lie on the polyline; uses min-distance edge
    snap to handle floating-point jitter."""
    edges_a = V_input
    edges_b = np.roll(V_input, -1, axis=0)
    edge_vecs = edges_b - edges_a
    edge_lens = np.linalg.norm(edge_vecs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(edge_lens)])

    rel = pts[:, None, :] - edges_a[None, :, :]                        # (M, n, 2)
    sq = edge_lens[None, :] ** 2 + 1e-30
    t = np.einsum("mij,ij->mi", rel, edge_vecs) / sq                   # (M, n)
    t_c = np.clip(t, 0.0, 1.0)
    proj = edges_a[None, :, :] + t_c[..., None] * edge_vecs[None, :, :]
    err = np.linalg.norm(pts[:, None, :] - proj, axis=-1)              # (M, n)
    best = err.argmin(axis=1)
    rows = np.arange(pts.shape[0])
    return cum[best] + t_c[rows, best] * edge_lens[best]


def test_arc_length_uniformity_on_irregular_circle():
    # 17-gon resampled to 64 points: euclidean segment lengths are NOT uniform
    # (corner-crossing segments shorten), but ARC-LENGTH positions along the
    # input polyline are exactly uniform.
    n = 17
    V0 = _circle(n, radius=1.0)
    V, L, nv = pad_polylines([V0], [_sequential_l(n)])

    M = 64
    V_new, _, _ = resample_uniform_polyline(V, L, nv, M)

    P = _perimeter(V0)
    arc = _arc_length_along(V0, V_new[0].numpy())
    expected = np.arange(M, dtype=np.float64) * P / M
    np.testing.assert_allclose(arc, expected, atol=1e-4)


def test_uniform_polygon_with_M_multiple_of_n():
    # When M is a multiple of n, every resampled point lies on a chord with no
    # corner-cutting -> euclidean segment lengths ARE uniform exactly.
    n = 17
    V0 = _circle(n, radius=1.0)
    V, L, nv = pad_polylines([V0], [_sequential_l(n)])
    M = n * 3
    V_new, _, _ = resample_uniform_polyline(V, L, nv, M)
    seg = _segment_lengths(V_new[0].numpy())
    assert seg.std() / seg.mean() < 1e-4


def test_square_points_lie_on_boundary():
    V0 = _square(side=2.0, per_edge=4)
    V, L, nv = pad_polylines([V0], [_sequential_l(V0.shape[0])])

    M = 40
    V_new, _, _ = resample_uniform_polyline(V, L, nv, M)

    expected_P = _perimeter(V0)
    got_P = _perimeter(V_new[0].numpy())
    assert abs(got_P - expected_P) / expected_P < 1e-4

    # Every resampled point must sit on one of the four sides.
    pts = V_new[0].numpy()
    half = 1.0
    dist_to_sides = np.minimum.reduce([
        np.abs(pts[:, 1] + half),
        np.abs(pts[:, 0] - half),
        np.abs(pts[:, 1] - half),
        np.abs(pts[:, 0] + half),
    ])
    assert dist_to_sides.max() < 1e-5

    # And arc-length uniformity.
    arc = _arc_length_along(V0, pts)
    expected = np.arange(M, dtype=np.float64) * expected_P / M
    np.testing.assert_allclose(arc, expected, atol=1e-4)


def test_first_resampled_vertex_equals_first_input_vertex():
    rng = np.random.default_rng(0)
    V0 = rng.standard_normal((23, 2)).astype(np.float32) * 1.5
    V, L, nv = pad_polylines([V0], [_sequential_l(23)])

    V_new, _, _ = resample_uniform_polyline(V, L, nv, 50)
    np.testing.assert_allclose(V_new[0, 0].numpy(), V0[0], atol=1e-5)


def test_canonical_L_and_nv_output():
    V0 = _circle(11)
    V, L, nv = pad_polylines([V0], [_sequential_l(11)])

    M = 7
    _, L_new, nv_new = resample_uniform_polyline(V, L, nv, M)
    expected_L = torch.tensor([[i, (i + 1) % M] for i in range(M)], dtype=torch.long)
    assert torch.equal(L_new[0], expected_L)
    assert nv_new.tolist() == [M]


def test_batched_variable_lengths():
    V_list = [_circle(13, radius=1.0), _square(side=2.0, per_edge=5), _circle(31, radius=0.5)]
    L_list = [_sequential_l(v.shape[0]) for v in V_list]
    V, L, nv = pad_polylines(V_list, L_list)

    M = 24
    V_new, L_new, nv_new = resample_uniform_polyline(V, L, nv, M)

    assert V_new.shape == (3, M, 2)
    assert L_new.shape == (3, M, 2)
    assert nv_new.tolist() == [M, M, M]

    for b, v0 in enumerate(V_list):
        np.testing.assert_allclose(V_new[b, 0].numpy(), v0[0], atol=1e-5)
        P = _perimeter(v0)
        arc = _arc_length_along(v0, V_new[b].numpy())
        expected = np.arange(M, dtype=np.float64) * P / M
        np.testing.assert_allclose(arc, expected, atol=1e-4)


def test_gradient_flows_through_V():
    V0 = _circle(20)
    V, L, nv = pad_polylines([V0], [_sequential_l(20)])
    V = V.detach().clone().requires_grad_(True)

    V_new, _, _ = resample_uniform_polyline(V, L, nv, 30)
    loss = (V_new ** 2).sum()
    loss.backward()

    assert V.grad is not None
    assert torch.isfinite(V.grad).all()
    assert V.grad[0].abs().sum().item() > 0


def test_idempotent_on_already_uniform_polygon():
    M = 24
    V0 = _circle(M, radius=2.0)
    V, L, nv = pad_polylines([V0], [_sequential_l(M)])
    V_new, _, _ = resample_uniform_polyline(V, L, nv, M)
    np.testing.assert_allclose(V_new[0].numpy(), V0, atol=1e-5)


def test_arc_length_uniformity_on_concave_shape():
    n_tips = 5
    n_pts = 2 * n_tips
    outer_r = 1.0
    inner_r = 0.45
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False, dtype=np.float32)
    radii = np.where(np.arange(n_pts) % 2 == 0, outer_r, inner_r).astype(np.float32)
    V0 = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    V, L, nv = pad_polylines([V0], [_sequential_l(n_pts)])
    M = 200
    V_new, _, _ = resample_uniform_polyline(V, L, nv, M)

    P = _perimeter(V0)
    arc = _arc_length_along(V0, V_new[0].numpy())
    expected = np.arange(M, dtype=np.float64) * P / M
    np.testing.assert_allclose(arc, expected, atol=1e-3)
