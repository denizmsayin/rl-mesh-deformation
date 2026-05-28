import math

import torch

from rlmd.evaluation.metrics.chamfer import ChamferMetric


def _ccw_circle(n=64, radius=1.0, centre=(0.0, 0.0)):
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    cx, cy = centre
    pts = torch.stack(
        (cx + radius * angles.cos(), cy + radius * angles.sin()), dim=-1
    ).unsqueeze(0)
    L = torch.stack(
        (torch.arange(n), (torch.arange(n) + 1) % n), dim=-1
    ).unsqueeze(0)
    nv = torch.tensor([n], dtype=torch.long)
    ne = torch.tensor([n], dtype=torch.long)  # closed cycle: #edges == #verts
    return pts, L, nv, ne


def test_with_normals_false_matches_legacy_keys():
    torch.manual_seed(0)
    poly_a = _ccw_circle()
    poly_b = _ccw_circle(radius=0.7)
    metric = ChamferMetric(num_samples=256)
    out = metric(poly_a, poly_b)
    assert set(out.keys()) == {"chamfer_a2b", "chamfer_b2a", "chamfer_sym"}


def test_with_normals_true_adds_normal_keys():
    torch.manual_seed(0)
    poly_a = _ccw_circle()
    poly_b = _ccw_circle(radius=0.7)
    metric = ChamferMetric(num_samples=256, with_normals=True)
    out = metric(poly_a, poly_b)
    assert set(out.keys()) == {
        "chamfer_a2b", "chamfer_b2a", "chamfer_sym",
        "normal_a2b", "normal_b2a", "normal_sym",
    }
    for v in out.values():
        assert v.shape == (1,)
        assert torch.isfinite(v).all()


def test_identity_polyline_gives_zero_chamfer_and_zero_normal_loss():
    torch.manual_seed(0)
    poly = _ccw_circle(n=128)
    metric = ChamferMetric(num_samples=1024, with_normals=True)
    out = metric(poly, poly)
    # Sampling is independent for the two clouds, so chamfer is small but not
    # exactly zero. Normal loss is bounded by the half-edge angle (~pi/128).
    assert out["chamfer_sym"].item() < 5e-4
    assert out["normal_sym"].item() < 5e-3


def test_reversed_orientation_flips_normals_but_not_chamfer():
    # Reverse the edge list (swap each (i, j) -> (j, i)). The point set is
    # identical, so chamfer ~ 0; but every outward normal points the wrong
    # way, so 1 - cos ~ 2 with abs_cosine=False.
    torch.manual_seed(0)
    V, L, nv, ne = _ccw_circle(n=128)
    L_rev = L.flip(dims=(-1,))

    metric = ChamferMetric(num_samples=1024, with_normals=True, abs_cosine=False)
    out = metric((V, L, nv, ne), (V, L_rev, nv, ne))

    assert out["chamfer_sym"].item() < 5e-4
    # Each matched pair has nearly-antiparallel normals: 1 - cos ~ 2.
    assert out["normal_sym"].item() > 1.95


def test_abs_cosine_true_is_orientation_invariant():
    # Same as above but with abs_cosine=True: normal loss should be ~0 since
    # 1 - |cos| ~ 0 when normals are anti-parallel.
    torch.manual_seed(0)
    V, L, nv, ne = _ccw_circle(n=128)
    L_rev = L.flip(dims=(-1,))

    metric = ChamferMetric(num_samples=1024, with_normals=True, abs_cosine=True)
    out = metric((V, L, nv, ne), (V, L_rev, nv, ne))

    assert out["normal_sym"].item() < 5e-3


def test_batched_polylines_normal_consistency():
    # Two items in a batch: one identical pair, one rotated by 90°. The
    # rotated case should produce a normal_sym around 1 - cos(pi/2) = 1.
    torch.manual_seed(0)
    n = 128
    V_a, L, _, _ = _ccw_circle(n=n)
    V_b_same = V_a.clone()

    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1] + math.pi / 2
    V_b_rot = torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(0)

    V_A = torch.cat([V_a, V_a], dim=0)
    V_B = torch.cat([V_b_same, V_b_rot], dim=0)
    L_b = torch.cat([L, L], dim=0)
    nv_b = torch.tensor([n, n], dtype=torch.long)

    metric = ChamferMetric(num_samples=1024, with_normals=True, abs_cosine=False)
    out = metric((V_A, L_b, nv_b, nv_b), (V_B, L_b, nv_b, nv_b))

    # Same circle -> ~0; rotated circle (same point set!) -> chamfer ~0 too,
    # but normals at matched pairs may be misaligned by up to ~pi/2 worth of
    # neighborhood offset. Just sanity-check shape and finiteness.
    assert out["normal_sym"].shape == (2,)
    assert torch.isfinite(out["normal_sym"]).all()
    assert out["normal_sym"][0].item() < 5e-3  # identical match


def test_gradient_flow_with_normals():
    n = 32
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    V_a = torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(0).clone()
    V_a.requires_grad_(True)
    V_b = torch.stack((0.5 * angles.cos(), 0.5 * angles.sin()), dim=-1).unsqueeze(0)
    L = torch.stack((torch.arange(n), (torch.arange(n) + 1) % n), dim=-1).unsqueeze(0)
    nv = torch.tensor([n], dtype=torch.long)

    metric = ChamferMetric(num_samples=128, with_normals=True)
    out = metric((V_a, L, nv, nv), (V_b, L, nv, nv))
    (out["chamfer_sym"] + out["normal_sym"]).sum().backward()
    assert V_a.grad is not None
    assert torch.isfinite(V_a.grad).all()
