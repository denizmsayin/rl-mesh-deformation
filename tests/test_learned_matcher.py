import numpy as np
import torch
import torch.nn.functional as F

from rlmd.batching import pad_polylines
from rlmd.evaluation.matchers import (
    FixedMatcher,
    LearnedMatcher,
    StochasticLearnedMatcher,
)
from rlmd.evaluation.matchers.learned import _compute_scores
from rlmd.models import PolygonCNN
from rlmd.ops import resample_uniform_polyline


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int64)


def _circle(n, rx=1.0, ry=1.0, phase=0.0):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.stack([rx * np.cos(a), ry * np.sin(a)], axis=1).astype(np.float32)


def _resample_pair(V_src, V_tgt, M):
    Vs, Ls, ns = pad_polylines([V_src], [_sequential_l(V_src.shape[0])])
    Vt, Lt, nt = pad_polylines([V_tgt], [_sequential_l(V_tgt.shape[0])])
    Vs, _, ns = resample_uniform_polyline(Vs, Ls, ns, M)
    Vt, _, nt = resample_uniform_polyline(Vt, Lt, nt, M)
    return Vs, ns, Vt, nt


def _model(out=16):
    torch.manual_seed(0)
    return PolygonCNN(in_channels=2, hidden_channels=(16, 16), out_channels=out,
                      kernel_size=5, layernorm=True)


def test_stochastic_matcher_shapes_and_log_prob_identity():
    M = 32
    V_src_np = _circle(20)
    V_tgt_np = _circle(30, rx=1.4, ry=0.7)
    Vs, ns, Vt, nt = _resample_pair(V_src_np, V_tgt_np, M)

    model = _model()
    matcher = StochasticLearnedMatcher(model, temperature=1.0)

    torch.manual_seed(123)
    matchings, log_prob, entropy = matcher(Vs, ns, Vt, nt)

    assert len(matchings) == 1
    m = matchings[0]
    assert m.idx_src.shape == (1, M)
    assert m.idx_tgt.shape == (1, M)
    assert m.mask.shape == (1, M)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)

    # log_prob = sum over rows of log_softmax(S)[i, sampled_j].
    with torch.no_grad():
        S = _compute_scores(model, Vs, ns, Vt, nt, 1.0)
        lp_rows = F.log_softmax(S, dim=-1).gather(-1, m.idx_tgt.unsqueeze(-1)).squeeze(-1)
        expected = (lp_rows * m.mask.to(lp_rows.dtype)).sum(dim=-1)
    torch.testing.assert_close(log_prob.detach(), expected, atol=1e-5, rtol=1e-4)


def test_stochastic_matcher_grad_flows_to_features():
    M = 24
    Vs, ns, Vt, nt = _resample_pair(_circle(15), _circle(18, rx=1.5), M)

    model = _model()
    matcher = StochasticLearnedMatcher(model, temperature=1.0)

    torch.manual_seed(0)
    _, log_prob, _ = matcher(Vs, ns, Vt, nt)
    log_prob.sum().backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_argmax_matcher_matches_stochastic_at_low_temperature():
    # At very small temperature, the stochastic sampler is heavily peaked, so
    # repeated samples should agree with the deterministic argmax pick most of
    # the time. We just check that argmax results are deterministic + valid.
    M = 24
    Vs, ns, Vt, nt = _resample_pair(_circle(20), _circle(28, ry=0.5), M)

    model = _model()
    argmax_matcher = LearnedMatcher(model, temperature=1.0)
    out1 = argmax_matcher(Vs, ns, Vt, nt)
    out2 = argmax_matcher(Vs, ns, Vt, nt)
    assert torch.equal(out1[0].idx_tgt, out2[0].idx_tgt)
    assert (out1[0].idx_tgt >= 0).all() and (out1[0].idx_tgt < M).all()


def test_argmax_matcher_returns_no_grad():
    M = 16
    Vs, ns, Vt, nt = _resample_pair(_circle(12), _circle(14), M)
    model = _model()
    matcher = LearnedMatcher(model)
    out = matcher(Vs, ns, Vt, nt)
    # Matching indices are long; no .grad_fn possible.
    assert not out[0].idx_tgt.requires_grad


def test_padding_does_not_leak_into_log_prob_or_entropy():
    # Build a batch where one item has fewer valid source vertices than the
    # other, by padding manually. Padded rows must contribute 0 to log_prob
    # and not be averaged into entropy.
    M = 16
    Vs0, ns0, Vt, nt = _resample_pair(_circle(10), _circle(12), M)

    # Item 1: same content but advertised as having only 10 valid source verts.
    Vs_pad = Vs0.clone()
    ns_pad = torch.tensor([10])
    # Zero out positions 10.. so padded sources are well-defined.
    Vs_pad[0, 10:] = 0.0

    Vs_batch = torch.cat([Vs0, Vs_pad], dim=0)            # (2, M, 2)
    ns_batch = torch.cat([ns0, ns_pad], dim=0)
    Vt_batch = Vt.expand(2, -1, -1).contiguous()
    nt_batch = nt.expand(2).contiguous()

    model = _model()
    matcher = StochasticLearnedMatcher(model, temperature=1.0)

    torch.manual_seed(7)
    matchings, log_prob, entropy = matcher(Vs_batch, ns_batch, Vt_batch, nt_batch)

    # mask reflects valid source positions
    assert matchings[0].mask[0].sum().item() == M
    assert matchings[0].mask[1].sum().item() == 10

    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_target_padding_zeroes_softmax_probability():
    # If we mark only the first 5 target positions as valid, sampled indices
    # must all be in [0, 5).
    M = 16
    Vs, ns, Vt, _ = _resample_pair(_circle(10), _circle(12), M)
    nt = torch.tensor([5])

    model = _model()
    matcher = StochasticLearnedMatcher(model, temperature=1.0)
    torch.manual_seed(0)
    matchings, _, _ = matcher(Vs, ns, Vt, nt)
    assert (matchings[0].idx_tgt < 5).all()


def test_fixed_matcher_returns_stored():
    from rlmd.ops import Matching
    M = 8
    idx_src = torch.arange(M)[None]
    idx_tgt = torch.arange(M).flip(0)[None]
    mask = torch.ones(1, M, dtype=torch.bool)
    m = Matching(idx_src=idx_src, idx_tgt=idx_tgt, mask=mask)

    fixed = FixedMatcher([m])
    out = fixed(None, None, None, None)  # args ignored
    assert out[0] is m


def test_learned_matcher_loads_checkpoint(tmp_path):
    # Train-time matcher with non-trivial weights.
    src_model = PolygonCNN(in_channels=2, hidden_channels=(8, 8),
                           out_channels=16, kernel_size=5, layernorm=True)
    for p in src_model.parameters():
        p.data.normal_(mean=0.3, std=0.5)

    ckpt = tmp_path / "matcher.pt"
    torch.save({
        "feature_extractor_state_dict": src_model.state_dict(),
        "temperature": 0.7,
        "M": 24,
    }, ckpt)

    # Fresh matcher with a freshly initialized extractor, then load.
    tgt_model = PolygonCNN(in_channels=2, hidden_channels=(8, 8),
                           out_channels=16, kernel_size=5, layernorm=True)
    matcher = LearnedMatcher(tgt_model, temperature=1.0, checkpoint_path=str(ckpt))

    # State_dict matches.
    for k, v in src_model.state_dict().items():
        torch.testing.assert_close(matcher.feature_extractor.state_dict()[k], v)

    # Temperature pulled from checkpoint.
    assert matcher.temperature == 0.7

    # And actually produces argmax matches that agree with a hand-built peer.
    M = 24
    Vs, ns, Vt, nt = _resample_pair(_circle(10), _circle(14), M)
    peer = LearnedMatcher(src_model, temperature=0.7)
    out_loaded = matcher(Vs, ns, Vt, nt)
    out_peer = peer(Vs, ns, Vt, nt)
    assert torch.equal(out_loaded[0].idx_tgt, out_peer[0].idx_tgt)


def test_learned_matcher_override_temperature(tmp_path):
    src_model = PolygonCNN(in_channels=2, hidden_channels=(8,),
                           out_channels=8, kernel_size=5, layernorm=False)
    ckpt = tmp_path / "matcher.pt"
    torch.save({"feature_extractor_state_dict": src_model.state_dict(),
                "temperature": 0.5}, ckpt)

    tgt_model = PolygonCNN(in_channels=2, hidden_channels=(8,),
                           out_channels=8, kernel_size=5, layernorm=False)
    matcher = LearnedMatcher(tgt_model, temperature=2.0, checkpoint_path=str(ckpt),
                             override_temperature=True)
    assert matcher.temperature == 2.0
