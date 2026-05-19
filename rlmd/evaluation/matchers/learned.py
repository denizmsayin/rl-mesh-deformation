from typing import List

import torch
import torch.nn.functional as F

from rlmd.ops import Matching


def _canonical_L(M: int, B: int, device) -> torch.Tensor:
    """Build canonical cyclic edges (i, (i+1) mod M) for a closed M-gon, batched."""
    i = torch.arange(M, device=device)
    L = torch.stack([i, (i + 1) % M], dim=-1)
    return L[None].expand(B, -1, -1).contiguous()


def _compute_scores(feature_extractor, V_src, n_src, V_tgt, n_tgt, temperature):
    """
    Returns (B, M_src, M_tgt) similarity logits with padded *target* positions
    masked to -inf so they get zero softmax probability.

    Assumes the input polylines are canonical cyclic sequences (true after
    `rlmd.ops.resample_uniform_polyline`); the canonical L is rebuilt here from
    num_verts so callers don't need to pass it.
    """
    B, M_src, _ = V_src.shape
    M_tgt = V_tgt.shape[1]

    L_src = _canonical_L(M_src, B, V_src.device)
    L_tgt = _canonical_L(M_tgt, B, V_tgt.device)

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

    def __call__(self, V_src, n_src, V_tgt, n_tgt):
        S = _compute_scores(self.feature_extractor, V_src, n_src, V_tgt, n_tgt,
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
    """

    name = "learned"

    def __init__(self, feature_extractor, temperature: float = 1.0):
        self.feature_extractor = feature_extractor
        self.temperature = float(temperature)

    def __call__(self, V_src, n_src, V_tgt, n_tgt) -> List[Matching]:
        with torch.no_grad():
            S = _compute_scores(self.feature_extractor, V_src, n_src, V_tgt, n_tgt,
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
