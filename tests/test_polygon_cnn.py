import numpy as np
import pytest
import torch

from rlmd.batching import pad_polylines
from rlmd.models import PolygonCNN, check_sequential_l, circular_pad_polygon


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)


def _random_polygon(n, rng):
    return rng.standard_normal((n, 2)).astype(np.float32), _sequential_l(n)


def test_circular_pad_wraps_valid_region():
    # Single item, no batch-level padding: pure circular wrap.
    n = 6
    pad = 2
    x = torch.arange(n, dtype=torch.float32).view(1, 1, n)
    out = circular_pad_polygon(x, torch.tensor([n]), pad)
    expected = torch.tensor([n - 2, n - 1, 0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.float32)
    assert out.shape == (1, 1, n + 2 * pad)
    assert torch.equal(out[0, 0], expected)


def test_circular_pad_ignores_batch_padding():
    # Two items with different n; the shorter one wraps over its own length only,
    # not over the batch padding zeros.
    n1, n2, N_max = 5, 3, 5
    pad = 2
    x = torch.zeros(2, 1, N_max)
    x[0, 0, :n1] = torch.tensor([10., 11., 12., 13., 14.])
    x[1, 0, :n2] = torch.tensor([20., 21., 22.])  # last two cols are batch padding
    out = circular_pad_polygon(x, torch.tensor([n1, n2]), pad)

    # Item 0 wraps over all 5 entries.
    expected_0 = torch.tensor([13., 14., 10., 11., 12., 13., 14., 10., 11.])
    assert torch.equal(out[0, 0], expected_0)

    # Item 1 wraps over its 3 entries only; positions past n2+pad-1 are zero.
    # ext positions [pad, pad+n2) = [2, 5) hold the valid run [20,21,22].
    # Left pad pulls (pos % 3): pos=-2 → 1 → 21; pos=-1 → 2 → 22.
    # Right pad pulls pos=3 → 0 → 20; pos=4 → 1 → 21. pos=5,6 are out of region → 0.
    expected_1 = torch.tensor([21., 22., 20., 21., 22., 20., 21., 0., 0.])
    assert torch.equal(out[1, 0], expected_1)


def test_check_sequential_l_passes_on_valid():
    n1, n2 = 7, 4
    rng = np.random.default_rng(0)
    V1, L1 = _random_polygon(n1, rng)
    V2, L2 = _random_polygon(n2, rng)
    _, L, num_verts, _ = pad_polylines([V1, V2], [L1, L2])
    # Should not raise.
    check_sequential_l(L, num_verts)


def test_check_sequential_l_raises_on_shuffled():
    n = 6
    rng = np.random.default_rng(1)
    V, L = _random_polygon(n, rng)
    perm = rng.permutation(n)
    L_bad = L[perm]
    _, L_pad, num_verts, _ = pad_polylines([V], [L_bad])
    with pytest.raises(ValueError):
        check_sequential_l(L_pad, num_verts)


def test_cnn_output_shape_and_masking():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    Vs, Ls = zip(*[_random_polygon(n, rng) for n in [10, 14, 7]])
    V, L, num_verts, _ = pad_polylines(list(Vs), list(Ls))

    model = PolygonCNN(in_channels=2, hidden_channels=(16, 32), out_channels=8, kernel_size=5)
    model.eval()
    out = model(V, L, num_verts, check_l=True)

    assert out.shape == (3, V.shape[1], 8)
    # Padded positions are zero.
    N_max = V.shape[1]
    pad_mask = torch.arange(N_max)[None, :] >= num_verts[:, None]
    assert torch.all(out[pad_mask] == 0)
    # At least one valid position is nonzero (sanity).
    valid_mask = ~pad_mask
    assert out[valid_mask].abs().sum() > 0


def test_cnn_unaffected_by_batch_padding():
    """A polygon's per-vertex features should be identical whether it sits in a
    batch alone or alongside a larger polygon that increases N_max."""
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    V_small, L_small = _random_polygon(8, rng)
    V_big, L_big = _random_polygon(20, rng)

    model = PolygonCNN(in_channels=2, hidden_channels=(16,), out_channels=4, kernel_size=5)
    model.eval()

    V_a, L_a, n_a, _ = pad_polylines([V_small], [L_small])
    out_a = model(V_a, L_a, n_a)

    V_b, L_b, n_b, _ = pad_polylines([V_small, V_big], [L_small, L_big])
    out_b = model(V_b, L_b, n_b)

    # Compare valid region of the small polygon across the two batches.
    n = V_small.shape[0]
    torch.testing.assert_close(out_a[0, :n], out_b[0, :n], atol=1e-6, rtol=1e-5)


def test_cnn_rotation_equivariant_along_cycle():
    """Cyclically rotating the vertex order of a single polygon should cyclically
    rotate the output features by the same offset."""
    torch.manual_seed(0)
    rng = np.random.default_rng(4)
    n = 12
    V_np, L_np = _random_polygon(n, rng)

    model = PolygonCNN(in_channels=2, hidden_channels=(16,), out_channels=4, kernel_size=5)
    model.eval()

    V, L, num_verts, _ = pad_polylines([V_np], [L_np])
    out = model(V, L, num_verts)

    shift = 4
    V_rot_np = np.roll(V_np, -shift, axis=0)
    V_rot, L_rot, n_rot, _ = pad_polylines([V_rot_np], [_sequential_l(n)])
    out_rot = model(V_rot, L_rot, n_rot)

    # out_rot[i] should equal out[(i + shift) % n].
    rolled = torch.roll(out[0, :n], shifts=-shift, dims=0)
    torch.testing.assert_close(out_rot[0, :n], rolled, atol=1e-5, rtol=1e-5)
