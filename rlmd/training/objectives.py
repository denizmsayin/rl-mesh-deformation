"""Policy-gradient objectives for matcher training.

Each objective owns one per-batch REINFORCE update: it calls the matcher,
rolls out the scenario, computes the reward, consults the baseline, and returns
the loss plus a flat metrics dict. This is the seam that keeps the RL semantics
out of ``scripts/train_matcher.py`` — the script just calls ``compute`` and
logs ``metrics``.

- `BanditObjective` — the original single-action problem: match once at t=0,
  freeze for the whole inner SGD, terminal reward, advantage ``R - b``.
- `RematchObjective` — re-match every ``rematch_every`` inner iters (a finite-
  horizon MDP). ``credit_mode``:
    * ``joint``  — terminal reward, advantage shared across the K matches,
      ``loss = -(R - b) · Σ_t log π(c_t)``. Cannot tell which match mattered.
    * ``dense``  — per-step return-to-go ``G_t = Φ_T - Φ_t`` (γ=1), advantage
      ``A_t``, ``loss = -Σ_t A_t · log π(c_t)``. With a `RolloutBaseline` this
      is multi-step SCST: ``A_t = Φ_T^sampled - Φ_T^greedy(from s_t)`` (the Φ_t
      terms cancel), giving per-match credit.

All metrics dicts share `METRIC_COLUMNS` so the CSV schema is objective-
agnostic and the existing training-curve plotter keeps working.
"""
from dataclasses import dataclass
from typing import Dict

import torch

from rlmd.evaluation.matchers import FixedMatcher
from rlmd.training.reward import CompositeChamferReward, Polyline

METRIC_COLUMNS = [
    "reward_mean", "reward_std",
    "reward_chamfer_mean", "reward_normal_mean",
    "advantage_mean", "advantage_std",
    "loss", "entropy", "baseline",
]


@dataclass
class Update:
    loss: torch.Tensor
    metrics: Dict[str, float]


def _std(x: torch.Tensor) -> float:
    return float(x.std().item()) if x.numel() > 1 else 0.0


def _metrics(R, advantage, loss, entropy, baseline_val,
             chamfer_sym, normal_sym) -> Dict[str, float]:
    return {
        "reward_mean": float(R.mean().item()),
        "reward_std": _std(R),
        "reward_chamfer_mean": float(chamfer_sym.mean().item()),
        "reward_normal_mean": float(normal_sym.mean().item()),
        "advantage_mean": float(advantage.mean().item()),
        "advantage_std": _std(advantage),
        "loss": float(loss.item()),
        "entropy": float(entropy.mean().item()),
        "baseline": float(baseline_val.mean().item()),
    }


class BanditObjective:
    """Single matching action, frozen for the inner SGD, terminal reward."""

    def __init__(self, scenario, reward: CompositeChamferReward, baseline,
                 entropy_coef: float = 0.0):
        self.scenario = scenario
        self.reward = reward
        self.baseline = baseline
        self.entropy_coef = float(entropy_coef)

    def compute(self, matcher, poly_src: Polyline, poly_tgt: Polyline) -> Update:
        V_src, L_src, nv_src = poly_src
        V_tgt, _, nv_tgt = poly_tgt

        matchings, log_prob, entropy = matcher(V_src, nv_src, V_tgt, nv_tgt)

        V_final = self.scenario.run(poly_src, poly_tgt, FixedMatcher(matchings))
        rout = self.reward((V_final, L_src, nv_src), poly_tgt)
        R = rout.reward                                              # (B,)

        if self.baseline.is_rollout:
            b = self.baseline.terminal_reward(poly_src, poly_tgt)    # (B,)
        else:
            b = self.baseline(R)                                     # (B,)

        advantage = (R - b).detach()
        loss = -(advantage * log_prob).mean() \
            - self.entropy_coef * entropy.mean()

        metrics = _metrics(R, advantage, loss, entropy, b,
                           rout.chamfer_sym, rout.normal_sym)
        return Update(loss=loss, metrics=metrics)


class RematchObjective:
    """Re-match every ``scenario.rematch_every`` inner iters; dense or joint."""

    def __init__(self, scenario, reward: CompositeChamferReward, baseline,
                 entropy_coef: float = 0.0, credit_mode: str = "dense"):
        if credit_mode not in ("dense", "joint"):
            raise ValueError(f"credit_mode must be 'dense' or 'joint', got {credit_mode!r}")
        self.scenario = scenario
        self.reward = reward
        self.baseline = baseline
        self.entropy_coef = float(entropy_coef)
        self.credit_mode = credit_mode

    def compute(self, matcher, poly_src: Polyline, poly_tgt: Polyline) -> Update:
        _, L_src, nv_src = poly_src

        rollout = self.scenario.run_policy(poly_src, poly_tgt, matcher)
        log_prob = rollout.log_prob          # (T, B)
        entropy = rollout.entropy            # (T, B)
        assert log_prob is not None and entropy is not None, (
            "RematchObjective needs a stochastic matcher (per-step log-probs)"
        )

        # Potential Φ at every boundary state s_0..s_T. The last boundary is
        # V_final, whose reward components we also log.
        phi_list, rout_final = [], None
        for t in range(rollout.boundary_V.shape[0]):
            rout = self.reward((rollout.boundary_V[t], L_src, nv_src), poly_tgt)
            phi_list.append(rout.reward)
            rout_final = rout
        phi = torch.stack(phi_list, dim=0)   # (T+1, B)
        phi_T = phi[-1]                      # (B,) — terminal reward R
        T = phi.shape[0] - 1

        if self.credit_mode == "joint":
            if self.baseline.is_rollout:
                b = self.baseline.terminal_reward(poly_src, poly_tgt)   # (B,)
            else:
                b = self.baseline(phi_T)                                # (B,)
            advantage = (phi_T - b).detach()                            # (B,)
            loss = -(advantage * log_prob.sum(dim=0)).mean() \
                - self.entropy_coef * entropy.mean()
            adv_for_log, b_for_log = advantage, b
        else:  # dense
            G = phi_T[None, :] - phi[:-1]                               # (T, B)
            if self.baseline.is_rollout:
                K = int(self.scenario.rematch_every)
                n = int(self.scenario.num_iters)
                refs = []
                for t in range(T):
                    rem = max(1, n - t * K)
                    refs.append(self.baseline.terminal_reward(
                        (rollout.boundary_V[t], L_src, nv_src), poly_tgt,
                        num_iters=rem))
                ref = torch.stack(refs, dim=0)                          # (T, B)
                # SCST: A_t = G_t - (ref_t - Φ_t) = Φ_T - ref_t  (Φ_t cancels).
                advantage = (phi_T[None, :] - ref).detach()            # (T, B)
                b_for_log = ref
            else:
                b_vec = self.baseline(G.mean(dim=0))                    # (B,)
                advantage = (G - b_vec[None, :]).detach()              # (T, B)
                b_for_log = b_vec
            loss = -(advantage * log_prob).sum(dim=0).mean() \
                - self.entropy_coef * entropy.mean()
            adv_for_log = advantage

        metrics = _metrics(phi_T, adv_for_log, loss, entropy, b_for_log,
                           rout_final.chamfer_sym, rout_final.normal_sym)
        return Update(loss=loss, metrics=metrics)
