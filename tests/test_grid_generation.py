import numpy as np
import torch

from rlmd.data.generation import Grid, Octagon, ShapeGenerator
from rlmd.batching import pad_polylines
from rlmd.ops import sample_points_from_polylines
from rlmd.evaluation.metrics import ChamferMetric


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


def test_octagon_is_axis_aligned():
    # The whole point of Octagon vs RegularPolygon(8): flat vertical/horizontal
    # sides, so a square grid can share full edges. A vertical side means two
    # vertices share an x-coordinate at the extreme right.
    pts = Octagon(num_points=80).get_points()
    x_max = pts[:, 0].max()
    on_right = np.isclose(pts[:, 0], x_max, atol=1e-6)
    assert on_right.sum() >= 2  # a flat vertical right side, not a single tip


def test_octagon_grid_welds_and_is_connected():
    g = Grid(num_points=480, cell_shape="octagon", rows=2, cols=2)
    pts = g.get_points()
    edges = g.get_edges()

    # Welding merged shared-side vertices, so fewer than the 4*per_cell raw points.
    assert pts.shape[0] < 4 * g._per_cell
    # Every edge references a valid welded vertex.
    assert edges.min() >= 0 and edges.max() < pts.shape[0]
    # The grid is a single connected graph (sharing full sides).
    assert _num_components(pts.shape[0], edges) == 1
    # The invariant the num_edges API exists for: #edges != #vertices.
    assert edges.shape[0] != pts.shape[0]


def test_circle_grid_welds_tangent_points_into_one_component():
    # Circles only touch at points; per_cell divisible by 4 lands a vertex on
    # each tangent point so welding still connects the four cells.
    g = Grid(num_points=480, cell_shape="circle", rows=2, cols=2)
    pts = g.get_points()
    edges = g.get_edges()
    assert _num_components(pts.shape[0], edges) == 1


def test_grid_registered_in_generator():
    gen = ShapeGenerator()
    base = gen.get_base_shape("grid", num_points=240, cell_shape="octagon", rows=2, cols=2)
    assert base.get_points().shape[0] > 0
    assert base.get_edges().shape[1] == 2


def test_grid_flows_through_sampling_and_chamfer():
    src = Grid(num_points=480, cell_shape="circle", rows=2, cols=2)
    tgt = Grid(num_points=480, cell_shape="octagon", rows=2, cols=2)

    V_s, L_s, nv_s, ne_s = pad_polylines([src.get_points()], [src.get_edges()])
    V_t, L_t, nv_t, ne_t = pad_polylines([tgt.get_points()], [tgt.get_edges()])

    # num_edges (not num_verts) drives the edge sampling.
    P = sample_points_from_polylines(V_s, L_s, ne_s, 500)
    assert P.shape == (1, 500, 2)
    assert torch.isfinite(P).all()

    out = ChamferMetric(num_samples=256)(
        (V_s, L_s, nv_s, ne_s), (V_t, L_t, nv_t, ne_t)
    )
    assert torch.isfinite(out["chamfer_sym"]).all()


def test_sampling_skips_the_central_hole():
    # The 2x2 octagon grid is centered at the origin and leaves a square hole
    # there (truncated-square tiling). Because we weld real boundary edges
    # instead of bridging across the gap, no sampled point should land in it.
    g = Grid(num_points=800, cell_shape="octagon", rows=2, cols=2)
    V, L, nv, ne = pad_polylines([g.get_points()], [g.get_edges()])
    P = sample_points_from_polylines(V, L, ne, 4000)[0]
    assert P.norm(dim=-1).min() > 0.15
