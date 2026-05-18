import math

import torch

from rlmd.ops import sample_points_from_polylines


def _closed_polyline(points):
    """Build (V, L, num_verts) for a single CCW closed polyline."""
    V = torch.as_tensor(points, dtype=torch.float32).unsqueeze(0)
    N = V.shape[1]
    L = torch.stack(
        (torch.arange(N), (torch.arange(N) + 1) % N),
        dim=-1,
    ).unsqueeze(0)
    nv = torch.tensor([N], dtype=torch.long)
    return V, L, nv


def test_default_return_unchanged():
    torch.manual_seed(0)
    angles = torch.linspace(0, 2 * math.pi, 32 + 1)[:-1]
    V, L, nv = _closed_polyline(torch.stack((angles.cos(), angles.sin()), dim=-1))
    out = sample_points_from_polylines(V, L, nv, num_samples=16)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 16, 2)


def test_returns_points_and_normals_with_right_shape():
    torch.manual_seed(0)
    angles = torch.linspace(0, 2 * math.pi, 64 + 1)[:-1]
    V, L, nv = _closed_polyline(torch.stack((angles.cos(), angles.sin()), dim=-1))
    pts, nrm = sample_points_from_polylines(V, L, nv, num_samples=128, return_normals=True)
    assert pts.shape == (1, 128, 2)
    assert nrm.shape == (1, 128, 2)


def test_normals_are_unit_length():
    torch.manual_seed(1)
    angles = torch.linspace(0, 2 * math.pi, 50 + 1)[:-1]
    V, L, nv = _closed_polyline(torch.stack((angles.cos(), angles.sin()), dim=-1))
    _, nrm = sample_points_from_polylines(V, L, nv, num_samples=256, return_normals=True)
    lens = nrm.norm(dim=-1)
    torch.testing.assert_close(lens, torch.ones_like(lens), atol=1e-5, rtol=1e-5)


def test_outward_normals_on_unit_circle():
    # On a CCW circle centred at origin, the outward normal at every sample
    # should point in the same direction as the sample's position vector.
    torch.manual_seed(2)
    angles = torch.linspace(0, 2 * math.pi, 256 + 1)[:-1]
    V, L, nv = _closed_polyline(torch.stack((angles.cos(), angles.sin()), dim=-1))
    pts, nrm = sample_points_from_polylines(V, L, nv, num_samples=512, return_normals=True)

    radial = pts / pts.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos = (radial * nrm).sum(dim=-1)
    # 256-gon: worst-case angle between sample-radial and edge-outward-normal is
    # half an edge, so cos >= cos(pi/256) ~ 0.99992. Loose bound is plenty.
    assert (cos > 0.999).all()


def test_outward_normals_on_ccw_square():
    # Square with vertices CCW: bottom, right, top, left edges should have
    # outward normals -y, +x, +y, -x.
    V = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    L = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]])
    nv = torch.tensor([4])
    expected = torch.tensor([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    torch.manual_seed(3)
    _, nrm = sample_points_from_polylines(V, L, nv, num_samples=2000, return_normals=True)

    # Check that every sampled normal matches one of the 4 expected edge normals.
    diffs = (nrm.squeeze(0).unsqueeze(1) - expected.unsqueeze(0)).norm(dim=-1)  # (S, 4)
    min_diff = diffs.min(dim=-1).values
    assert (min_diff < 1e-5).all()
    # And we should actually hit all four edges with non-trivial probability.
    hit = diffs.argmin(dim=-1)
    assert hit.unique().numel() == 4


def test_gradients_flow_through_normals():
    torch.manual_seed(4)
    angles = torch.linspace(0, 2 * math.pi, 40 + 1)[:-1]
    pts0 = torch.stack((angles.cos(), angles.sin()), dim=-1)
    V = pts0.unsqueeze(0).clone().requires_grad_(True)
    N = V.shape[1]
    L = torch.stack(
        (torch.arange(N), (torch.arange(N) + 1) % N),
        dim=-1,
    ).unsqueeze(0)
    nv = torch.tensor([N], dtype=torch.long)

    _, nrm = sample_points_from_polylines(V, L, nv, num_samples=64, return_normals=True)
    nrm.sum().backward()
    assert V.grad is not None
    assert torch.isfinite(V.grad).all()


def test_padded_batch_normals_finite():
    # Two items in a batch with different vertex counts (padded). Padded edges
    # have zero length so multinomial never picks them; the eps clamp keeps
    # things finite regardless.
    angles_a = torch.linspace(0, 2 * math.pi, 16 + 1)[:-1]
    pts_a = torch.stack((angles_a.cos(), angles_a.sin()), dim=-1)

    angles_b = torch.linspace(0, 2 * math.pi, 8 + 1)[:-1]
    pts_b = torch.stack((2 * angles_b.cos(), 2 * angles_b.sin()), dim=-1)

    N_max = 16
    V = torch.zeros(2, N_max, 2)
    V[0] = pts_a
    V[1, :8] = pts_b

    L = torch.zeros(2, N_max, 2, dtype=torch.long)
    L[0] = torch.stack(
        (torch.arange(N_max), (torch.arange(N_max) + 1) % N_max),
        dim=-1,
    )
    closed_b = torch.stack(
        (torch.arange(8), (torch.arange(8) + 1) % 8),
        dim=-1,
    )
    L[1, :8] = closed_b  # pad rows stay (0, 0): degenerate self-loops.

    nv = torch.tensor([16, 8])

    torch.manual_seed(5)
    pts, nrm = sample_points_from_polylines(V, L, nv, num_samples=128, return_normals=True)
    assert torch.isfinite(pts).all()
    assert torch.isfinite(nrm).all()
    torch.testing.assert_close(
        nrm.norm(dim=-1), torch.ones(2, 128), atol=1e-5, rtol=1e-5
    )
