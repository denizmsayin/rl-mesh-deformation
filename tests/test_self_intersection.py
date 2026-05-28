import math

import torch

from rlmd.evaluation.metrics.self_intersection import SelfIntersectionMetric


def _poly(V, L, n):
    """Build a (V, L, num_verts, num_edges) polyline tuple for the metric API."""
    ne = (L >= 0).all(dim=-1).sum(dim=-1).long()
    return (V, L, n, ne)


def _closed_polygon(pts):
    """Build (V, L, n) for a single closed polygon from a list of (x, y)."""
    V = torch.tensor(pts, dtype=torch.float32).unsqueeze(0)
    n = V.shape[1]
    L = torch.stack(
        (torch.arange(n), (torch.arange(n) + 1) % n), dim=-1
    ).unsqueeze(0)
    return V, L, torch.tensor([n], dtype=torch.long)


def test_name_and_output_shape():
    metric = SelfIntersectionMetric()
    assert metric.name == "self_intersection"
    V, L, n = _closed_polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    out = metric(_poly(V, L, n), _poly(V, L, n))
    assert set(out.keys()) == {"self_intersection"}
    assert out["self_intersection"].shape == (1,)


def test_simple_square_has_no_self_intersections():
    V, L, n = _closed_polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    out = SelfIntersectionMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["self_intersection"].item() == 0.0


def test_figure_eight_has_one_self_intersection():
    # Bowtie / figure-eight: square with two diagonally-opposite vertices
    # swapped in the visiting order. Edges (0,1) and (2,3) cross once.
    V, L, n = _closed_polygon([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
    out = SelfIntersectionMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["self_intersection"].item() == 1.0


def test_pentagram_has_five_self_intersections():
    # Star polygon {5/2}: connect every second vertex of a regular pentagon.
    angles = torch.linspace(0, 2 * math.pi, 6)[:-1] + math.pi / 2
    pts = torch.stack((angles.cos(), angles.sin()), dim=-1)
    order = torch.tensor([0, 2, 4, 1, 3])
    pts = pts[order]
    V = pts.unsqueeze(0)
    L = torch.stack(
        (torch.arange(5), (torch.arange(5) + 1) % 5), dim=-1
    ).unsqueeze(0)
    n = torch.tensor([5])
    out = SelfIntersectionMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["self_intersection"].item() == 5.0


def test_ccw_circle_has_no_self_intersections():
    n = 64
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    V = torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(0)
    L = torch.stack(
        (torch.arange(n), (torch.arange(n) + 1) % n), dim=-1
    ).unsqueeze(0)
    out = SelfIntersectionMetric()(_poly(V, L, torch.tensor([n])), _poly(V, L, torch.tensor([n])))
    assert out["self_intersection"].item() == 0.0


def test_padding_is_ignored():
    # A clean square padded with -1 edges and an extra unused vertex slot.
    V = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [99.0, 99.0],  # padding vertex, unreferenced via num_verts
    ]])
    L = torch.tensor([[
        [0, 1], [1, 2], [2, 3], [3, 0],
        [-1, -1], [-1, -1],
    ]])
    n = torch.tensor([4])
    out = SelfIntersectionMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["self_intersection"].item() == 0.0


def test_out_of_range_edges_are_ignored():
    # A bowtie's crossing edge is "out of range" via num_verts=2 -> count = 0.
    V = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ]])
    L = torch.tensor([[
        [0, 1], [1, 2], [2, 3], [3, 0],
    ]])
    n_full = torch.tensor([4])
    n_clip = torch.tensor([2])  # only edges referencing verts {0,1} are valid
    out_full = SelfIntersectionMetric()(_poly(V, L, n_full), _poly(V, L, n_full))
    out_clip = SelfIntersectionMetric()(_poly(V, L, n_clip), _poly(V, L, n_clip))
    assert out_full["self_intersection"].item() == 0.0
    assert out_clip["self_intersection"].item() == 0.0


def test_batched_independence():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    bowtie = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    V = torch.tensor([square, bowtie], dtype=torch.float32)
    L = torch.stack(
        (torch.arange(4), (torch.arange(4) + 1) % 4), dim=-1
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    n = torch.tensor([4, 4])
    out = SelfIntersectionMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["self_intersection"].tolist() == [0.0, 1.0]


def test_shared_endpoint_is_not_counted():
    # Degenerate polyline 0->1->0->2 that revisits vertex 0; edges (1,0) and
    # (0,2) share vertex 0. Touching at the shared point is not a proper
    # crossing, so count must be 0.
    V = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    ]])
    L = torch.tensor([[
        [0, 1], [1, 0], [0, 2],
    ]])
    n = torch.tensor([3])
    out = SelfIntersectionMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["self_intersection"].item() == 0.0
