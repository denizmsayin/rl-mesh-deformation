import numpy as np
import torch

from rlmd.batching import pad_polylines
from rlmd.data.generation import Grid
from rlmd.models import PolygonGNN


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int64)


def _random_polygon(n, rng):
    return rng.standard_normal((n, 2)).astype(np.float32), _sequential_l(n)


def test_output_shape_and_padding_zeroed():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Vs, Ls = zip(*[_random_polygon(n, rng) for n in [10, 14, 7]])
    V, L, nv, ne = pad_polylines(list(Vs), list(Ls))

    model = PolygonGNN(in_channels=2, hidden_channels=(16, 32), out_channels=8)
    model.eval()
    out = model(V, L, nv, ne)

    assert out.shape == (3, V.shape[1], 8)
    N_max = V.shape[1]
    pad_mask = torch.arange(N_max)[None, :] >= nv[:, None]
    assert torch.all(out[pad_mask] == 0)
    assert out[~pad_mask].abs().sum() > 0


def test_num_edges_optional_matches_explicit():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    Vs, Ls = zip(*[_random_polygon(n, rng) for n in [9, 13]])
    V, L, nv, ne = pad_polylines(list(Vs), list(Ls))

    model = PolygonGNN(in_channels=2, hidden_channels=(16,), out_channels=4)
    model.eval()
    out_explicit = model(V, L, nv, ne)
    out_derived = model(V, L, nv)  # num_edges inferred from L
    torch.testing.assert_close(out_explicit, out_derived)


def test_unaffected_by_batch_padding():
    """A graph's per-vertex features must be identical whether it sits alone or
    alongside a larger graph that increases N_max / M_max."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    V_small, L_small = _random_polygon(8, rng)
    V_big, L_big = _random_polygon(20, rng)

    model = PolygonGNN(in_channels=2, hidden_channels=(16,), out_channels=4)
    model.eval()

    V_a, L_a, nv_a, ne_a = pad_polylines([V_small], [L_small])
    out_a = model(V_a, L_a, nv_a, ne_a)

    V_b, L_b, nv_b, ne_b = pad_polylines([V_small, V_big], [L_small, L_big])
    out_b = model(V_b, L_b, nv_b, ne_b)

    n = V_small.shape[0]
    torch.testing.assert_close(out_a[0, :n], out_b[0, :n], atol=1e-6, rtol=1e-5)


def test_permutation_equivariance():
    """Relabelling vertices (and remapping edges) permutes the outputs by the
    same permutation. This is the property the CNN faked via sequential order."""
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    n = 12
    V_np, L_np = _random_polygon(n, rng)

    model = PolygonGNN(in_channels=2, hidden_channels=(16,), out_channels=4)
    model.eval()

    V, L, nv, ne = pad_polylines([V_np], [L_np])
    out = model(V, L, nv, ne)

    perm = rng.permutation(n)
    inv = np.argsort(perm)
    V_perm = V_np[perm]                       # new index i holds old vertex perm[i]
    L_perm = inv[L_np]                        # remap endpoints to new indices
    Vp, Lp, nvp, nep = pad_polylines([V_perm], [L_perm.astype(np.int64)])
    out_perm = model(Vp, Lp, nvp, nep)

    # out_perm[i] should equal out[perm[i]].
    torch.testing.assert_close(out_perm[0, :n], out[0, perm], atol=1e-5, rtol=1e-5)


def test_runs_on_welded_grid():
    g = Grid(num_points=480, cell_shape="octagon", rows=2, cols=2)
    V_np = g.get_points().astype(np.float32)
    L_np = g.get_edges().astype(np.int64)
    V, L, nv, ne = pad_polylines([V_np], [L_np])

    model = PolygonGNN(in_channels=2, hidden_channels=(32, 32), out_channels=16)
    model.eval()
    out = model(V, L, nv, ne)

    n = V_np.shape[0]
    assert out.shape == (1, n, 16)
    assert torch.isfinite(out).all()
    assert out[0, :n].abs().sum() > 0


def test_gradient_flows():
    rng = np.random.default_rng(4)
    V_np, L_np = _random_polygon(15, rng)
    V, L, nv, ne = pad_polylines([V_np], [L_np])
    V = V.detach().clone().requires_grad_(True)

    model = PolygonGNN(in_channels=2, hidden_channels=(16,), out_channels=8)
    out = model(V, L, nv, ne)
    out.sum().backward()

    assert V.grad is not None
    assert torch.isfinite(V.grad).all()
    assert V.grad.abs().sum().item() > 0
