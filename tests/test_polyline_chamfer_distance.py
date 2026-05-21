import torch
from pytorch3d.loss import chamfer_distance

from rlmd.ops import distance_loss, knn_match


def test_self_distance_is_zero():
    torch.manual_seed(0)
    P = torch.randn(2, 100, 2)
    n = torch.tensor([100, 80])
    matchings = knn_match(P, n, P, n, bidirectional=True)
    assert distance_loss(P, P, matchings, p=2).item() < 1e-10


def test_matches_pytorch3d_chamfer():
    # pytorch3d chamfer is (dir1 + dir2); ours is (dir1 + dir2) / 2.
    # We sum over batch; pytorch3d means. So 2 * ours / B == ref.
    torch.manual_seed(0)
    B, N_s, N_t, D = 3, 120, 150, 2
    P_src = torch.randn(B, N_s, D)
    P_tgt = torch.randn(B, N_t, D)
    n_src = torch.tensor([120, 100, 80])
    n_tgt = torch.tensor([150, 90, 140])

    matchings = knn_match(P_src, n_src, P_tgt, n_tgt, bidirectional=True)
    ours = distance_loss(P_src, P_tgt, matchings, p=2)

    ref, _ = chamfer_distance(
        P_src, P_tgt,
        x_lengths=n_src, y_lengths=n_tgt,
        batch_reduction='mean', point_reduction='mean',
    )
    torch.testing.assert_close(2 * ours / B, ref)


def test_unidirectional_returns_single_matching():
    torch.manual_seed(0)
    P_src = torch.randn(2, 30, 2)
    P_tgt = torch.randn(2, 40, 2)
    n_src = torch.tensor([30, 20])
    n_tgt = torch.tensor([40, 30])
    matchings = knn_match(P_src, n_src, P_tgt, n_tgt, bidirectional=False)
    assert len(matchings) == 1
    # Loss should run and be finite.
    loss = distance_loss(P_src, P_tgt, matchings, p=2)
    assert torch.isfinite(loss)


def test_gradient_flows_through_src():
    torch.manual_seed(0)
    P_src = torch.randn(1, 50, 2, requires_grad=True)
    P_tgt = torch.randn(1, 60, 2)
    n_src = torch.tensor([50])
    n_tgt = torch.tensor([60])
    matchings = knn_match(P_src, n_src, P_tgt, n_tgt, bidirectional=True)
    loss = distance_loss(P_src, P_tgt, matchings, p=2)
    loss.backward()
    assert P_src.grad is not None
    assert torch.isfinite(P_src.grad).all()
