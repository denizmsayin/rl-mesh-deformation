# Initial design: learned REINFORCE matcher

Working notes for training a learned matcher (PolygonCNN features + dot-product
similarity, sampled actions, REINFORCE updates) to replace the
nearest-neighbour matcher in the existing harness.

This is a discussion document, not a spec. Decisions still open are listed at
the bottom.

## Context recap

- Current harness: `dataset_src` x `dataset_tgt` paired positionally; for each
  pair a `Matcher` produces `List[Matching]` (discrete index pairs between two
  padded point clouds) and a `Scenario` consumes those matches inside a SGD
  loop with Chamfer + edge + Laplacian + normal-consistency terms.
- `PolygonCNN` already exists in `rlmd/models/polygon_cnn.py` and gives
  per-vertex features with circular-padded 1D conv.
- `Matcher` protocol is in `rlmd/evaluation/matchers/base.py`; only `Knn3dMatcher`
  exists today. A learned matcher that returns the same `List[Matching]` will
  drop into the existing harness with zero changes at eval time.

## Why one-step is not a useful regime

Quick literature / intuition check, since the proposal mentions a one-step
proxy and Deniz's Chamfer experiments suggest it doesn't actually fit
anything:

- Chamfer / NN-matching gradients form a *velocity field*, not a target
  field. After one step toward `V_tgt[C]` you either overshoot or collapse
  many sources onto one target. Iteration is intrinsic — Voxel2Mesh /
  Pixel2Mesh / AtlasNet / MeshSDF all use multi-step residual deformation
  with regularizers active throughout.
- Regularization is local — Laplacian smoothing propagates one ring per
  step, so corners/details need O(N) iterations to develop.
- The matching's value shows up across the trajectory, not at one step. A
  one-step reward throws away the only mechanism by which the matching
  could help.

Conclusion: drop Stage A. The one-step formulation survives only as a
gradient-flow smoke test, not a real regime.

## Two stages

### Stage 1 — Match-once + frozen-match SGD

- Sample matches once at t=0 from the learned policy.
- Inner SGD runs for `num_iters` with the **fixed** correspondences as the
  data term: `‖P_t[Cᵢ_src] − P_t_tgt[Cᵢ_tgt]‖²`, plus existing edge /
  Laplacian / normal-consistency regularizers.
- Reward at end of trajectory, single REINFORCE update.

Important: the data term must be the fixed-pair loss, **not** Chamfer
recomputed each iter. Otherwise the policy only sets initial conditions for
an optimizer that ignores it afterwards.

Cost note: Deniz's Chamfer experiments need ~1000+ iters for a reasonable
fit. Two mitigations:

- **Train on a shorter horizon (~200 iters) than we evaluate on (~1000+).**
  The matching's value is most visible in the early-to-mid trajectory; once
  SGD has run 1000 iters it has often masked out poor initial matches
  anyway.
- Large batch (B ≥ 128) on GPU. Inner SGD has no policy gradients flowing
  through it (matches are fixed indices), so it's a deterministic batched
  ODE — cheap per-step but amortizes well.

### Stage 2 — Re-match every K iters

Multi-step RL. K ≈ 20–50 keeps the trajectory short enough for REINFORCE
without GAE. Only attempted if Stage 1 shows a positive signal.

### Reward

Candidates:

- **R = −Chamfer(V_final, V_tgt)**. Direct. Chamfer is fine as a *scalar
  reward* even though it is a bad continuous gradient — gradients only flow
  through log π.
- **R = Chamfer_NN_baseline − Chamfer_policy**. "Did the learned matcher beat
  NN on this pair?" Centers the advantage near zero. Baseline can be
  **precomputed offline** over the dataset and cached.
- **Normalized improvement**: (Chamfer_init − Chamfer_final) / Chamfer_init.
  Scale-invariant across hard/easy pairs.
- **Ground-truth correspondence supervised loss** (warm-start, not RL). Our
  procedural shapes are parameterized, so the "ideal" correspondence is often
  known (arc-length / angle).
- **Shape-quality augmentation**: add normal-consistency / self-intersection
  count to the reward to discourage collapse.

Starting recommendation: cached NN-baseline advantage + entropy bonus.
Optional supervised pretraining for variance reduction.

### Action structure

DISK-style: S = F_src · F_tgtᵀ / τ → (B, M, N). For each source point i,
sample j ∼ Cat(softmax(S[i,·])). Joint log-prob = Σᵢ log π(jᵢ|i). Argmax at
eval.

Match space: **grid-sampled points along arc length** (not vertices, not
random samples).

- Vertex-only fails on shapes with very few vertices (e.g. stars with sharp
  corners). Sampling equalizes point counts across shapes.
- Random sampling injects noise into the action distribution every iter
  (different sampled points → different features → different optimal
  matches), fighting the policy. Deterministic uniform arc-length sampling
  gives the policy a stable point cloud.
- The action space is then **grid index → grid index**, which stays
  meaningful across inner-SGD iterations even as V moves, because the
  parameterization is arc-length on whichever V is current.

Feature extraction:

- Run PolygonCNN on **vertices** (where the topology / circular conv
  lives).
- Interpolate features to grid points via segment-linear interpolation.
- Do not run a separate point-cloud net on the grid points — loses
  topology.

Variants still open:

- Unidirectional vs bidirectional (matches existing bidirectional Chamfer).
- Subsample K << M anchors per pair (proposal: "subset of matches as control
  points").
- Entropy regularization with schedulable temperature τ.

## Architecture sketch

- Reuse `PolygonCNN` + a small projection head → `D_match`.
- New `rlmd/matchers/learned.py`: `LearnedMatcher` with `mode ∈
  {"sample","argmax"}`. Sample mode returns `(Matching, log_prob, entropy)`;
  argmax mode returns `Matching` so the existing harness eval path works.
- New scenario for Stage B: `rlmd/evaluation/scenarios/sgd_fixed_match.py` —
  SGD with frozen correspondences as the data term.
- New `scripts/train_matcher.py` + `configs/train_matcher.yaml`. Hydra
  instantiation for model, optimizer, env mode (`one_step` / `frozen_sgd`),
  reward type, baseline source.
- Cache file for NN baseline reward, keyed by dataset_src/dataset_tgt names.
- Eval: drop trained matcher in argmax mode into existing harness, sweep
  against `knn_3d`.

## Things to be careful about

- Don't re-sample edge points each iter inside the matcher. Match on
  vertices; sampling stays at the reward-Chamfer stage.
- Frozen-match SGD ≠ re-matching with Chamfer. The policy must own the data
  term throughout the trajectory in Stage B.
- The proposal's "300+ episodes/sec on CPU" target only holds for Stage A.
  Stage B amortizes only with large batch (B ≥ 64).

## Decisions still open

1. Reward: cached NN-baseline advantage / plain neg Chamfer / normalized
   improvement.
2. Pretrain with parametric ground-truth correspondences (yes/no).
3. Action sampling: all M unidirectional / all M bidirectional / K anchors.
4. Training horizon vs eval horizon (proposed: 200 vs 1000+).
5. Number of grid-sampled points M (proposed: 64 or 128, matching current
   `num_samples=500` may be overkill for the action space).
