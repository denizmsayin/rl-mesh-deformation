import math

import torch

from rlmd.evaluation.metrics.segment_std import SegmentStdMetric


def _poly(V, L, n):
    """Build a (V, L, num_verts, num_edges) polyline tuple for the metric API."""
    ne = (L >= 0).all(dim=-1).sum(dim=-1).long()
    return (V, L, n, ne)


def _segments_std_loop(points, connect, lengths, unbiased=False):
    """Reference loop implementation mirroring the original segments_std."""
    B = points.shape[0]
    per_batch = []
    for b in range(B):
        l = int(lengths[b].item())
        pts_b = points[b, :l, :]
        seg_lens = []
        for segment in connect[b]:
            i = int(segment[0].item())
            j = int(segment[1].item())
            if 0 <= i < l and 0 <= j < l:
                seg_lens.append(torch.norm(pts_b[i] - pts_b[j]))
        if len(seg_lens) > 0:
            sl = torch.stack(seg_lens)
            if unbiased and sl.numel() < 2:
                per_batch.append(points.new_tensor(0.0))
            else:
                per_batch.append(sl.std(unbiased=unbiased))
        else:
            per_batch.append(points.new_tensor(0.0))
    return torch.stack(per_batch)


def test_returns_per_sample_vector_and_name():
    metric = SegmentStdMetric()
    assert metric.name == "segment_std"
    V = torch.randn(3, 5, 2)
    L = torch.tensor([
        [[0, 1], [1, 2], [2, 0], [-1, -1], [-1, -1]],
        [[0, 1], [1, 2], [2, 3], [3, 0], [-1, -1]],
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
    ])
    n = torch.tensor([3, 4, 5])
    out = metric(_poly(V, L, n), _poly(V, L, n))
    assert set(out.keys()) == {"segment_std"}
    assert out["segment_std"].shape == (3,)
    assert torch.isfinite(out["segment_std"]).all()


def test_matches_loop_reference_with_padding():
    torch.manual_seed(0)
    B, N, M, D = 4, 10, 12, 2
    V = torch.randn(B, N, D)
    L = torch.randint(0, N, (B, M, 2))
    L[:, -3:] = -1  # tail-pad edges
    # truncate valid vertex count for some items so that some real edges
    # reference vertex indices that are now out of range — exercising the
    # masking logic.
    n = torch.tensor([N, N - 1, N - 3, N])
    ref = _segments_std_loop(V, L, n, unbiased=False)
    out = SegmentStdMetric()(_poly(V, L, n), _poly(V, L, n))
    torch.testing.assert_close(out["segment_std"], ref)


def test_uniform_polygon_has_zero_std():
    angles = torch.linspace(0.0, 2.0 * math.pi, 6)[:-1]
    V = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).unsqueeze(0)
    L = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]])
    n = torch.tensor([5])
    out = SegmentStdMetric()(_poly(V, L, n), _poly(V, L, n))
    assert out["segment_std"].shape == (1,)
    assert out["segment_std"].item() < 1e-6


def test_no_valid_edges_returns_zero():
    V = torch.randn(2, 3, 2)
    L = torch.full((2, 4, 2), -1, dtype=torch.long)
    n = torch.tensor([3, 3])
    out = SegmentStdMetric()(_poly(V, L, n), _poly(V, L, n))
    torch.testing.assert_close(out["segment_std"], torch.zeros(2))


def test_unbiased_flag_matches_loop():
    torch.manual_seed(1)
    V = torch.randn(2, 6, 2)
    L = torch.tensor([
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        [[0, 1], [1, 2], [2, 3], [3, 0], [-1, -1]],
    ])
    n = torch.tensor([6, 4])
    out = SegmentStdMetric(unbiased=True)(_poly(V, L, n), _poly(V, L, n))
    ref = _segments_std_loop(V, L, n, unbiased=True)
    torch.testing.assert_close(out["segment_std"], ref)


def test_known_value_two_segments():
    # batch of one polyline with 3 collinear vertices on x-axis at 0, 1, 3.
    # segments: (0-1) length 1, (1-2) length 2. population std = 0.5.
    V = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    L = torch.tensor([[[0, 1], [1, 2]]])
    n = torch.tensor([3])
    out = SegmentStdMetric()(_poly(V, L, n), _poly(V, L, n))
    torch.testing.assert_close(out["segment_std"], torch.tensor([0.5]))


def test_independent_across_batch_items():
    # changing the second item must not change the first item's value.
    torch.manual_seed(2)
    V1 = torch.randn(1, 5, 2)
    L1 = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]])
    n1 = torch.tensor([5])
    single = SegmentStdMetric()(_poly(V1, L1, n1), _poly(V1, L1, n1))["segment_std"]

    V2 = torch.randn(1, 5, 2) * 100.0  # very different scale
    V = torch.cat([V1, V2], dim=0)
    L = torch.cat([L1, L1], dim=0)
    n = torch.cat([n1, n1], dim=0)
    batched = SegmentStdMetric()(_poly(V, L, n), _poly(V, L, n))["segment_std"]
    torch.testing.assert_close(batched[:1], single)


def test_gradient_flows_through_vertices():
    V = torch.randn(2, 5, 2, requires_grad=True)
    L = torch.tensor([
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[0, 1], [1, 2], [2, 3], [-1, -1]],
    ])
    n = torch.tensor([5, 4])
    out = SegmentStdMetric()(_poly(V, L, n), _poly(V, L, n))
    out["segment_std"].sum().backward()
    assert V.grad is not None
    assert torch.isfinite(V.grad).all()
