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

## The three coupled design axes

These choices aren't independent — pick one and the rest narrows down.

### 1. Match frequency

| | episode = | actions per ep | credit assignment | proposal stage |
|---|---|---|---|---|
| **A. Match-once + one-step move** | 1 step | M (vertices) | trivial — single action vector | one-step proxy |
| **B. Match-once + frozen-match SGD** | full SGD with fixed Cₜ | M | terminal reward, still 1 action | bridge to multi-step |
| **C. Re-match every K iters** | T/K steps | M·(T/K) | discounted / GAE | true RL |

A → B → C, gated on results. C as Stage 1 will drown in variance.

Important nuance for B: the data term during SGD must be the fixed-pair loss
`‖V[i] − V_tgt[Cᵢ]‖²`, **not** Chamfer recomputed each iter. Otherwise the
policy only sets the initial conditions for an optimizer that ignores it
afterwards, and there is almost no signal.

Open question (raised by Deniz): is Stage A meaningful at all? Chamfer
experiments suggest one step is insufficient to reach any reasonable shape.
See the discussion section.

### 2. Reward

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

### 3. Action structure

DISK-style: S = F_src · F_tgtᵀ / τ → (B, M, N). For each source vertex i,
sample j ∼ Cat(softmax(S[i,·])). Joint log-prob = Σᵢ log π(jᵢ|i). Argmax at
eval.

Variants:

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

## Open: is Stage A viable?

Deniz's Chamfer experiments suggest a single deformation step cannot reach a
reasonable shape, which makes a one-step reward both noisy and uninformative.
Possible adjustments:

- Reframe "one step" as **N closed-form line-pulls** with a small step size
  per matched pair, plus K Laplacian smoothing sweeps — still no SGD, but more
  than a single move. Cheap and deterministic given C.
- Skip A entirely and start with B, but cap `num_iters` (e.g. 100–300) to keep
  episodes fast. This is what Deniz's Chamfer baselines effectively assume.
- Hybrid: A's deformation operator = one gradient descent step on the
  fixed-match data term + regularizers, large lr. Gives a single-step
  trajectory but uses the same machinery as B.

## Decisions still open

1. Starting stage: A (one-step), B (frozen-match SGD), or both.
2. Reward: cached NN-baseline advantage / plain neg Chamfer / normalized
   improvement.
3. Pretrain with parametric ground-truth correspondences (yes/no).
4. Action sampling: all M unidirectional / all M bidirectional / K anchors.
5. (New, from this discussion) What deformation operator do we use in Stage A
   if we keep A at all?
