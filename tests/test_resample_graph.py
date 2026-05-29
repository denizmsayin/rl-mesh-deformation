import numpy as np
import pytest
import torch

from rlmd.batching import pad_polylines
from rlmd.data.generation import Grid
from rlmd.ops import resample_uniform_graph, resample_uniform_polyline


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int64)


def _circle(n, radius=1.0, phase=0.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)


def _square(side=1.0, per_edge=8):
    s, half = side, side / 2.0
    t = np.linspace(0.0, 1.0, per_edge, endpoint=False, dtype=np.float32)
    edges = [
        np.stack([-half + t * s, np.full_like(t, -half)], axis=1),
        np.stack([np.full_like(t, half), -half + t * s], axis=1),
        np.stack([half - t * s, np.full_like(t, half)], axis=1),
        np.stack([np.full_like(t, -half), half - t * s], axis=1),
    ]
    return np.concatenate(edges, axis=0)


def _num_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(int(a))] = find(int(b))
    return len({find(i) for i in range(n)})


def _grid_batch(cell, k=1, num_points=480):
    g = Grid(num_points=num_points, cell_shape=cell, rows=2, cols=2)
    V_np = g.get_points().astype(np.float32)
    L_np = g.get_edges().astype(np.int64)
    return pad_polylines([V_np] * k, [L_np] * k), V_np, L_np


# --- backward compatibility: pure cycles reduce to resample_uniform_polyline ---

def test_reduces_to_polyline_on_single_cycle():
    V0 = _circle(17, radius=1.0)
    V, L, nv, ne = pad_polylines([V0], [_sequential_l(17)])
    M = 64
    g = resample_uniform_graph(V, L, nv, ne, M)
    p = resample_uniform_polyline(V, L, nv, M)
    for a, b in zip(g, p):
        torch.testing.assert_close(a, b)


def test_reduces_to_polyline_on_variable_length_cycle_batch():
    V_list = [_circle(13), _square(side=2.0, per_edge=5), _circle(31, radius=0.5)]
    L_list = [_sequential_l(v.shape[0]) for v in V_list]
    V, L, nv, ne = pad_polylines(V_list, L_list)
    M = 24
    g = resample_uniform_graph(V, L, nv, ne, M)
    p = resample_uniform_polyline(V, L, nv, M)
    for a, b in zip(g, p):
        torch.testing.assert_close(a, b)


# --- grid graphs ---

def test_circle_grid_pins_junctions_and_stays_connected():
    (V, L, nv, ne), V_np, L_np = _grid_batch("circle")
    n = V_np.shape[0]
    deg = np.zeros(n, dtype=int)
    for a, b in L_np:
        deg[a] += 1
        deg[b] += 1
    junctions = np.where(deg != 2)[0]
    assert len(junctions) == 4

    M = 64
    V_new, L_new, nv_new, ne_new = resample_uniform_graph(V, L, nv, ne, M)
    assert V_new.shape == (1, M, 2)
    assert int(nv_new[0]) == M

    # Every input junction position must appear exactly in the output (pinned).
    out = V_new[0].numpy()
    for j in junctions:
        d = np.linalg.norm(out - V_np[j][None, :], axis=1)
        assert d.min() < 1e-5, f"junction {j} not pinned in output"

    # Output graph is a single connected component.
    out_edges = L_new[0, : int(ne_new[0])].numpy()
    assert _num_components(M, out_edges) == 1


def test_octagon_grid_resamples_connected():
    (V, L, nv, ne), V_np, _ = _grid_batch("octagon")
    M = 96
    V_new, L_new, nv_new, ne_new = resample_uniform_graph(V, L, nv, ne, M)
    assert V_new.shape == (1, M, 2)
    assert torch.isfinite(V_new).all()
    out_edges = L_new[0, : int(ne_new[0])].numpy()
    assert _num_components(M, out_edges) == 1


def test_batched_homogeneous_grid_pins_each_item():
    # Three identical-topology grids at different poses share one plan but each
    # item's junctions must pin to its own coordinates.
    g = Grid(num_points=480, cell_shape="circle", rows=2, cols=2)
    L_np = g.get_edges().astype(np.int64)
    V0 = g.get_points().astype(np.float32)
    V1 = (V0 * 2.0 + 1.0).astype(np.float32)
    V2 = (V0[:, ::-1] * 0.5).astype(np.float32)
    V, L, nv, ne = pad_polylines([V0, V1, V2], [L_np, L_np, L_np])

    M = 64
    V_new, _, _, _ = resample_uniform_graph(V, L, nv, ne, M)
    assert V_new.shape == (3, M, 2)

    deg = np.zeros(V0.shape[0], dtype=int)
    for a, b in L_np:
        deg[a] += 1
        deg[b] += 1
    junctions = np.where(deg != 2)[0]
    for item, Vsrc in enumerate([V0, V1, V2]):
        out = V_new[item].numpy()
        for j in junctions:
            d = np.linalg.norm(out - Vsrc[j][None, :], axis=1)
            assert d.min() < 1e-4


def test_warns_on_short_chain_at_small_M():
    (V, L, nv, ne), _, _ = _grid_batch("circle")
    # 4 junctions, 8 arcs; M=12 -> 8 free verts / 8 arcs = 1 each -> 2 segments < 3.
    with pytest.warns(UserWarning, match="fewer than"):
        resample_uniform_graph(V, L, nv, ne, 12)


def test_no_warning_when_chains_long_enough():
    import warnings

    (V, L, nv, ne), _, _ = _grid_batch("circle")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resample_uniform_graph(V, L, nv, ne, 80)


def test_heterogeneous_graph_batch_raises():
    g1 = Grid(num_points=480, cell_shape="circle", rows=2, cols=2)
    g2 = Grid(num_points=480, cell_shape="octagon", rows=2, cols=2)
    V, L, nv, ne = pad_polylines(
        [g1.get_points().astype(np.float32), g2.get_points().astype(np.float32)],
        [g1.get_edges().astype(np.int64), g2.get_edges().astype(np.int64)],
    )
    with pytest.raises(ValueError):
        resample_uniform_graph(V, L, nv, ne, 64)


def test_gradient_flows_through_graph_resample():
    (V, L, nv, ne), _, _ = _grid_batch("circle")
    V = V.detach().clone().requires_grad_(True)
    V_new, _, _, _ = resample_uniform_graph(V, L, nv, ne, 80)
    V_new.pow(2).sum().backward()
    assert V.grad is not None
    assert torch.isfinite(V.grad).all()
    assert V.grad.abs().sum().item() > 0
