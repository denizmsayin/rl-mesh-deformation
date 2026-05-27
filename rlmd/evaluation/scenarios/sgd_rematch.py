from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from rlmd.evaluation.scenarios._inner import _inner_loss_compiled

Polyline = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (V, L, num_verts)


@dataclass
class RematchRollout:
    """Result of a re-matching rollout, for multi-step REINFORCE.

    Fields:
        V_final:    (B, M, 2) final deformed source vertices.
        boundary_V: (T+1, B, M, 2) source vertices at each re-match boundary,
                    i.e. the state s_t that decision c_t was taken from, plus
                    the final configuration. The objective computes the
                    potential Φ(s_t) from these to form per-step rewards.
        log_prob:   (T, B) joint log-prob of the matching sampled at each
                    boundary, or None for a deterministic matcher.
        entropy:    (T, B) mean per-item entropy at each boundary, or None.
        frames:     optional (num_frames, K, M, 2) CPU snapshots for viz.
    """

    V_final: torch.Tensor
    boundary_V: torch.Tensor
    log_prob: Optional[torch.Tensor]
    entropy: Optional[torch.Tensor]
    frames: Optional[torch.Tensor] = None


@dataclass
class SgdRematchScenario:
    """
    SGD deformation that RE-MATCHES every ``rematch_every`` iterations.

    Generalizes `SgdFixedMatchScenario` from a single frozen correspondence to
    a sequence of them: the inner loop runs in segments of ``rematch_every``
    iters, and at the start of each segment the matcher is called on the
    *current* deformed vertices ``(V_src + deform).detach()`` (and ``V_tgt``),
    producing a fresh matching that is then frozen for that segment.

    This turns the bandit (one matching action, terminal reward) into a
    finite-horizon MDP with ``T = ceil(num_iters / rematch_every)`` decision
    points. The state at boundary ``t`` is the current deformed shape; the
    matcher is the policy. See `rlmd.training.objectives.RematchObjective`.

    Same vertex-direct data term and regularizers as `SgdFixedMatchScenario`
    (shared `_inner_loss`). Inputs are assumed already uniformly resampled.

    Gradient note: the matcher is called on detached vertices, so no gradient
    flows to policy params through the inner SGD. REINFORCE puts the policy
    gradient on the returned `log_prob`, which is held by the caller.
    """

    name: str = "sgd_rematch"
    num_iters: int = 200
    rematch_every: int = 40
    lr: float = 1.0
    momentum: float = 0.9
    w_data: float = 1.0
    w_edge: float = 1.0
    w_normal: float = 0.01
    w_laplacian: float = 0.1
    distance_p: int = 2

    def run_policy(
        self,
        poly_src: Polyline,
        poly_tgt: Polyline,
        matcher,
        *,
        record_every: Optional[int] = None,
        record_max_batch: Optional[int] = None,
    ) -> RematchRollout:
        V_src, L_src, nv_src = poly_src
        V_tgt, _, nv_tgt = poly_tgt

        deform = torch.zeros_like(V_src, requires_grad=True)
        optimizer = torch.optim.SGD([deform], lr=self.lr, momentum=self.momentum)

        K = int(self.rematch_every)
        n = int(self.num_iters)

        boundary: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        frames = [] if record_every is not None else None
        rec_b = record_max_batch if record_max_batch is not None else V_src.shape[0]

        def _snapshot(V_now: torch.Tensor) -> None:
            frames.append(V_now[:rec_b].detach().to("cpu", copy=True))

        global_iter = 0
        i = 0
        while i < n:
            seg = min(K, n - i)

            V_cur = (V_src + deform).detach().clone()
            boundary.append(V_cur)

            res = matcher(V_cur, nv_src, V_tgt, nv_tgt)
            if isinstance(res, tuple):                  # stochastic policy
                matchings, log_p, ent = res
                log_probs.append(log_p)
                entropies.append(ent)
            else:                                        # deterministic (argmax/knn)
                matchings = res

            for _ in range(seg):
                optimizer.zero_grad()
                total, V = _inner_loss_compiled(
                    V_src, deform, V_tgt, matchings, L_src, nv_src,
                    self.w_data, self.w_edge, self.w_normal, self.w_laplacian,
                    self.distance_p,
                )
                if frames is not None and global_iter % record_every == 0:
                    _snapshot(V)
                total.backward()
                optimizer.step()
                global_iter += 1

            i += seg

        V_final = (V_src + deform).detach()
        boundary.append(V_final)
        if frames is not None:
            _snapshot(V_final)

        return RematchRollout(
            V_final=V_final,
            boundary_V=torch.stack(boundary, dim=0),
            log_prob=torch.stack(log_probs, dim=0) if log_probs else None,
            entropy=torch.stack(entropies, dim=0) if entropies else None,
            frames=torch.stack(frames, dim=0) if frames is not None else None,
        )

    def run(
        self,
        poly_src: Polyline,
        poly_tgt: Polyline,
        matcher,
        *,
        record_every: Optional[int] = None,
        record_max_batch: Optional[int] = None,
    ):
        """Matcher-protocol entry: returns V_final (or the recording 3-tuple).

        Used by eval / baseline rollouts with a deterministic matcher. Training
        calls `run_policy` directly to get per-step log-probs.
        """
        rollout = self.run_policy(
            poly_src, poly_tgt, matcher,
            record_every=record_every, record_max_batch=record_max_batch,
        )
        if record_every is not None:
            return rollout.V_final, rollout.frames, None
        return rollout.V_final
