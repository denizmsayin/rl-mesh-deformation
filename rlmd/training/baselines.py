"""REINFORCE baselines for matcher training.

Two families, both consumed uniformly by the objectives:

- `ScalarBaseline` — state-independent running statistic (none / ema). Returns
  a per-batch broadcast value. Replicates the previous inline ``_Baseline``.
- `RolloutBaseline` — state-conditional reference: roll a frozen/greedy policy
  from a given start state to the end and score its terminal Φ. Covers the old
  ``prior`` (SCST under the frozen-at-init policy) and ``chamfer_sgd`` (KNN
  re-matching SGD) baselines, which differ only in which (scenario, matcher)
  they roll out. The dense multi-step objective queries it once per re-match
  boundary, shortening the horizon via ``num_iters``.

``build_baseline`` maps a ``configs/baseline/*.yaml`` block onto the right one.
"""
import copy
from dataclasses import replace
from typing import Optional

import torch
from hydra.utils import instantiate

from rlmd.evaluation.matchers import LearnedMatcher
from rlmd.training.reward import CompositeChamferReward, Polyline


class ScalarBaseline:
    """State-independent baseline over a fed scalar series (none / ema).

    Call with the per-item rewards ``R`` (B,); returns the baseline value to
    subtract, broadcast to ``R``'s shape. For ``ema`` the returned value is the
    pre-update running mean (first batch falls back to the batch mean so the
    very first advantage is centered), then the EMA is updated with this batch.
    """

    is_rollout = False

    def __init__(self, kind: str = "none", momentum: float = 0.99):
        self.kind = str(kind)
        self.name = self.kind
        if self.kind == "ema":
            self.momentum = float(momentum)
            self.value: Optional[float] = None
        elif self.kind != "none":
            raise ValueError(f"unknown scalar-baseline kind: {self.kind!r}")

    def __call__(self, R: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return torch.zeros_like(R)
        batch_mean = R.detach().mean().item()
        b = batch_mean if self.value is None else self.value
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * batch_mean
        return torch.full_like(R, b)


class RolloutBaseline:
    """State-conditional baseline: terminal Φ of a greedy rollout from a state.

    ``terminal_reward(poly_src, poly_tgt)`` runs ``scenario.run`` from the given
    start vertices with ``matcher`` (a deterministic argmax / KNN matcher) and
    returns the per-item reward Φ of the result. ``num_iters`` shortens the
    rollout horizon — used by the dense objective so the reference from boundary
    ``t`` covers exactly the iterations that remain in the sampled episode.
    """

    is_rollout = True

    def __init__(self, scenario, matcher, reward: CompositeChamferReward,
                 name: str = "rollout"):
        self.scenario = scenario
        self.matcher = matcher
        self.reward = reward
        self.name = name

    def terminal_reward(self, poly_src: Polyline, poly_tgt: Polyline,
                        num_iters: Optional[int] = None) -> torch.Tensor:
        scen = (self.scenario if num_iters is None
                else replace(self.scenario, num_iters=int(num_iters)))
        V_final = scen.run(poly_src, poly_tgt, self.matcher)
        _, L_src, nv_src = poly_src
        return self.reward((V_final, L_src, nv_src), poly_tgt).reward


def build_baseline(baseline_cfg, *, training_scenario, reward: CompositeChamferReward,
                   feature_extractor, temperature: float):
    """Construct the baseline object from a ``configs/baseline/*`` block.

    ``feature_extractor`` must be the *uncompiled* trainable extractor; for the
    ``prior`` baseline it is deep-copied and frozen here, so this must run
    BEFORE the trainable extractor is wrapped in ``torch.compile`` (otherwise
    the prior would share weight tensors with the policy).
    """
    btype = str(baseline_cfg.get("type", "none"))
    if btype in ("none", "ema"):
        return ScalarBaseline(kind=btype,
                              momentum=float(baseline_cfg.get("momentum", 0.99)))
    if btype == "prior":
        prior_extractor = copy.deepcopy(feature_extractor)
        for p in prior_extractor.parameters():
            p.requires_grad = False
        matcher = LearnedMatcher(torch.compile(prior_extractor), temperature=temperature)
        return RolloutBaseline(scenario=training_scenario, matcher=matcher,
                               reward=reward, name="prior")
    if btype == "chamfer_sgd":
        scenario = instantiate(baseline_cfg.scenario)
        matcher = instantiate(baseline_cfg.matcher)
        return RolloutBaseline(scenario=scenario, matcher=matcher,
                               reward=reward, name="chamfer_sgd")
    raise ValueError(f"unknown baseline type: {btype!r}")
