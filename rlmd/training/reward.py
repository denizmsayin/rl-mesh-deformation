"""Composite Chamfer reward used as the RL potential Φ.

Φ(V) = -(w_chamfer * chamfer_sym + w_normal * normal_sym) between a deformed
source polyline and the target. This is the single source of truth for the
reward across objectives and baselines: the bandit terminal reward, the
per-state baseline rollouts, and the dense per-step shaping reward all call it.

Kept deliberately small and stateless so it can be shared by reference.
"""
from dataclasses import dataclass
from typing import Tuple

import torch

from rlmd.evaluation.metrics import ChamferMetric

Polyline = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # (V, L, num_verts, num_edges)


@dataclass
class RewardOut:
    """Per-sample reward plus the components that feed it (for logging)."""

    reward: torch.Tensor       # (B,) — Φ
    chamfer_sym: torch.Tensor  # (B,)
    normal_sym: torch.Tensor   # (B,)


class CompositeChamferReward:
    """Callable Φ over a pair of (V, L, num_verts, num_edges) polylines.

    Wraps a ``ChamferMetric(with_normals=True)``. Always runs under
    ``no_grad`` — the reward only ever feeds a detached advantage, so it never
    needs to participate in the policy-gradient graph.
    """

    def __init__(self, num_samples: int = 8192, w_chamfer: float = 1.0,
                 w_normal: float = 0.0):
        self.num_samples = int(num_samples)
        self.w_chamfer = float(w_chamfer)
        self.w_normal = float(w_normal)
        self._chamfer = ChamferMetric(
            num_samples=self.num_samples,
            point_reduction="mean",
            norm=2,
            with_normals=True,
        )

    @torch.no_grad()
    def __call__(self, poly_src: Polyline, poly_tgt: Polyline) -> RewardOut:
        out = self._chamfer(poly_src, poly_tgt)
        reward = -(self.w_chamfer * out["chamfer_sym"]
                   + self.w_normal * out["normal_sym"])
        return RewardOut(
            reward=reward,
            chamfer_sym=out["chamfer_sym"],
            normal_sym=out["normal_sym"],
        )
