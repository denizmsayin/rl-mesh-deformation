from typing import List, Optional

import torch
import torch.nn.functional as F

from rlmd.ops import Matching


def _compute_scores(feature_extractor, poly_src, poly_tgt, temperature):
    """
    Returns (B, M_src, M_tgt) similarity logits with padded *target* positions
    masked to -inf so they get zero softmax probability.

    Consumes the real polyline graphs: the feature extractor is given each
    side's actual (V, L, num_verts), so topology beyond a single cycle (e.g.
    welded grids) is preserved. The extractor is called with 3 positional args,
    which both PolygonCNN (ignores L past an optional check) and PolygonGNN
    (derives num_edges from L) accept.
    """
    V_src, L_src, n_src, _ = poly_src
    V_tgt, L_tgt, n_tgt, _ = poly_tgt
    M_tgt = V_tgt.shape[1]

    f_src = feature_extractor(V_src, L_src, n_src)            # (B, M_src, D)
    f_tgt = feature_extractor(V_tgt, L_tgt, n_tgt)            # (B, M_tgt, D)

    S = torch.bmm(f_src, f_tgt.transpose(1, 2)) / temperature  # (B, M_src, M_tgt)

    tgt_valid = (torch.arange(M_tgt, device=V_tgt.device)[None, :] < n_tgt[:, None])
    S = S.masked_fill(~tgt_valid[:, None, :], float("-inf"))
    return S


def _src_valid_mask(M_src: int, n_src: torch.Tensor) -> torch.Tensor:
    return torch.arange(M_src, device=n_src.device)[None, :] < n_src[:, None]


class StochasticLearnedMatcher:
    """
    Samples one target index per source point from softmax(S / tau).

    Returns the sampled correspondences along with the joint log-probability
    (sum over valid source positions) and the mean per-item entropy, for
    REINFORCE training.

    Signature is the same as the deterministic Matcher protocol's args but
    returns a 3-tuple, so it is not a drop-in replacement for that protocol.
    Wrap a sampled Matching in `FixedMatcher` before passing it to a scenario.
    """

    name = "learned_stochastic"

    def __init__(self, feature_extractor, temperature: float = 1.0):
        self.feature_extractor = feature_extractor
        self.temperature = float(temperature)

    def __call__(self, poly_src, poly_tgt):
        V_src, n_src = poly_src[0], poly_src[2]
        S = _compute_scores(self.feature_extractor, poly_src, poly_tgt,
                            self.temperature)
        B, M_src, _ = S.shape

        log_p = F.log_softmax(S, dim=-1)                            # (B, M_src, M_tgt)
        dist = torch.distributions.Categorical(logits=S)
        sample = dist.sample()                                       # (B, M_src), no grad

        src_valid = _src_valid_mask(M_src, n_src)
        gathered = log_p.gather(-1, sample.unsqueeze(-1)).squeeze(-1)  # (B, M_src)
        gathered = gathered * src_valid.to(gathered.dtype)
        log_prob = gathered.sum(dim=-1)                              # (B,)

        ent_row = dist.entropy() * src_valid.to(S.dtype)             # (B, M_src)
        denom = src_valid.sum(dim=-1).clamp(min=1).to(S.dtype)
        entropy = ent_row.sum(dim=-1) / denom                        # (B,)

        idx_src = torch.arange(M_src, device=V_src.device)[None, :].expand(B, -1)
        matching = Matching(idx_src=idx_src, idx_tgt=sample, mask=src_valid)
        return [matching], log_prob, entropy


class LearnedMatcher:
    """
    Deterministic argmax wrapper around a feature extractor. Conforms to the
    existing Matcher protocol (returns List[Matching]) so it slots into
    scenarios and the harness eval path unchanged.

    When ``checkpoint_path`` is given, the feature extractor's weights are
    loaded from a checkpoint produced by ``scripts/train_matcher.py``
    (key ``feature_extractor_state_dict``). The stored ``temperature``, if
    present, overrides the ``temperature`` kwarg unless
    ``override_temperature=True``.
    """

    name = "learned"

    def __init__(
        self,
        feature_extractor,
        temperature: float = 1.0,
        checkpoint_path: Optional[str] = None,
        override_temperature: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.temperature = float(temperature)
        if checkpoint_path is not None:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.feature_extractor.load_state_dict(ck["feature_extractor_state_dict"])
            if "temperature" in ck and not override_temperature:
                self.temperature = float(ck["temperature"])
        self.feature_extractor.eval()

    def __call__(self, poly_src, poly_tgt) -> List[Matching]:
        V_src, n_src = poly_src[0], poly_src[2]
        self.feature_extractor.to(V_src.device)
        with torch.no_grad():
            S = _compute_scores(self.feature_extractor, poly_src, poly_tgt,
                                self.temperature)
        sample = S.argmax(dim=-1)                                    # (B, M_src)
        B, M_src = sample.shape
        idx_src = torch.arange(M_src, device=V_src.device)[None, :].expand(B, -1)
        src_valid = _src_valid_mask(M_src, n_src)
        return [Matching(idx_src=idx_src, idx_tgt=sample, mask=src_valid)]


class FixedMatcher:
    """
    Adapter that returns a pre-computed Matching list. Used in the RL training
    loop to hand a sampled action to a scenario while keeping log_prob held
    separately upstream.
    """

    name = "fixed"

    def __init__(self, matchings: List[Matching]):
        self._matchings = matchings

    def __call__(self, *args, **kwargs) -> List[Matching]:
        return self._matchings
