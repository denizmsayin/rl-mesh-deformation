import math

import torch

from data_generation.generate import ShapeGenerator
from rlmd.batching import pad_polylines
from rlmd.ops import (
    polyline_edge_loss,
    polyline_laplacian_smoothing,
    polyline_normal_consistency,
)


def _circle_batch(radii=(0.5, 1.0, 1.5), num_points=60):
    gen = ShapeGenerator()
    shapes = [gen.build_shape('circle', num_points=num_points, scale=s) for s in radii]
    return pad_polylines([s.get_points() for s in shapes], [s.get_edges() for s in shapes])


def test_edge_loss_matches_circle_analytic():
    # For a circle of radius r with n vertices: edge length = 2r sin(pi/n).
    # With target=0, per-shape mean squared edge length = 4 r^2 sin^2(pi/n).
    # Batch mean across radii: 4 sin^2(pi/n) * mean(r^2).
    radii = (0.5, 1.0, 1.5)
    n = 60
    V, L, nv = _circle_batch(radii=radii, num_points=n)

    expected = 4 * math.sin(math.pi / n) ** 2 * sum(r ** 2 for r in radii) / len(radii)
    torch.testing.assert_close(polyline_edge_loss(V, L, nv), torch.tensor(expected))


def test_laplacian_matches_circle_analytic():
    # For a circle: ||Delta v_i|| = r (1 - cos(2*pi/n)), uniform across vertices.
    # Batch mean: mean(r) * (1 - cos(2*pi/n)).
    radii = (0.5, 1.0, 1.5)
    n = 60
    V, L, nv = _circle_batch(radii=radii, num_points=n)

    expected = (sum(radii) / len(radii)) * (1 - math.cos(2 * math.pi / n))
    torch.testing.assert_close(polyline_laplacian_smoothing(V, L, nv), torch.tensor(expected))


def test_normal_consistency_matches_circle_analytic():
    # For a circle: turning angle at each vertex = 2*pi/n, scale-invariant.
    # Per-vertex loss = 1 - cos(2*pi/n); batch mean is the same.
    radii = (0.5, 1.0, 1.5)
    n = 60
    V, L, nv = _circle_batch(radii=radii, num_points=n)

    expected = 1 - math.cos(2 * math.pi / n)
    torch.testing.assert_close(polyline_normal_consistency(V, L, nv), torch.tensor(expected))


def test_normal_consistency_triangle_sharp_corners():
    # Bare triangle (3 vertices): each corner turns by 2*pi/3, so per-vertex
    # loss = 1 - cos(2*pi/3) = 1.5.
    gen = ShapeGenerator()
    tri = gen.build_shape('triangle', num_points=3)
    V, L, nv = pad_polylines([tri.get_points()], [tri.get_edges()])

    expected = 1 - math.cos(2 * math.pi / 3)
    torch.testing.assert_close(polyline_normal_consistency(V, L, nv), torch.tensor(expected))
