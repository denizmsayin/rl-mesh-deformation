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
fit.

- Make `num_iters_train` and `num_iters_eval` separate config knobs.
- **Start with both equal** for simplicity (no proxy / eval gap to reason
  about).
- Once Stage 1 works, we may train on a shorter horizon (~100–200) than we
  eval on, since the matching's value is most visible early in the
  trajectory.
- Large batch (B ≥ 128) on GPU. Inner SGD has no policy gradients flowing
  through it (matches are fixed indices), so it's a deterministic batched
  ODE — cheap per-step but amortizes well.

### Stage 2 — Re-match every K iters

Multi-step RL. K ≈ 20–50 keeps the trajectory short enough for REINFORCE
without GAE. Only attempted if Stage 1 shows a positive signal.

### Reward

**Start with: R = −Chamfer(V_final, V_tgt)** with an EMA scalar baseline for
variance reduction. Chamfer is fine as a *scalar reward* even though it's a
bad continuous gradient — gradients only flow through log π, not through
Chamfer's NN matching.

Future plans (not Stage 1):

- **Cached NN-baseline advantage**: R = Chamfer_NN − Chamfer_policy with the
  NN-rollout Chamfer precomputed offline per (src, tgt) pair. Centers
  advantage near zero. Add once Stage 1 is stable to reduce variance further.
- **Normalized improvement**: (Chamfer_init − Chamfer_final) / Chamfer_init.
  Scale-invariant across hard/easy pairs.
- **Shape-quality augmentation**: add normal-consistency / self-intersection
  count to discourage collapse.
- **Ground-truth correspondence supervised loss** as a warm-start.

### Action structure

DISK-style: S = F_src · F_tgtᵀ / τ → (B, M, M). For each source point i,
sample j ∼ Cat(softmax(S[i,·])). Joint log-prob = Σᵢ log π(jᵢ|i). Argmax at
eval.

**Uniform resampling to a new polyline.** Rather than sampling points off
the original variable-vertex polyline, resample each shape *once per
episode* to a uniform-stride, fixed-count polyline with M vertices spaced
equally along arc length. Treat that as the actual source/target mesh from
then on.

Consequences:

- Action space is exactly **M × M** vertex-to-vertex matching.
- PolygonCNN feeds directly on the resampled polyline — no feature
  interpolation needed.
- The CNN's receptive field covers a constant arc-length per kernel
  position across all shapes, which makes the convolution semantically
  consistent.
- The inner SGD optimizes the resampled V (not the original variable-vertex
  V).
- Eval Chamfer can still sample further from the optimized polyline; that
  stays the existing harness behaviour.
- A possible loss is fine resolution on sharp corners (stars). Mitigation:
  M large enough that arc-length resolution > corner scale; in practice
  M ∈ {64, 128} should suffice for our procedural shapes.

Implementation: a `resample_uniform_polyline(V, L, num_verts, M)` op,
applied either as a dataset transform or at episode reset. Dataset
transform is cleaner because then `num_verts` is constant and
batching/padding is trivial.

Variants still open:

- Unidirectional vs bidirectional matching.
- Subsample K << M anchors per pair (proposal: "subset of matches as
  control points").
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

## Decisions made

- **Reward**: plain negative final Chamfer + EMA scalar baseline.
- **Horizons**: configurable `num_iters_train` / `num_iters_eval`; start
  equal.
- **Point representation**: resample to uniform-stride M-vertex polyline,
  used as the actual source/target mesh.
- **M**: configurable, default **128**.
- **No supervised pretrain.** Pure RL from scratch — the experimental
  question is whether RL works at all on this setup, not whether it beats
  arc-length-parameterization-with-extra-help.

## Decisions still open

1. Action sampling: independent per-row softmax sampling vs enforced
   one-to-one (Plackett-Luce / sequential sampling without replacement).
   See discussion below.
2. Unidirectional vs bidirectional matching.
