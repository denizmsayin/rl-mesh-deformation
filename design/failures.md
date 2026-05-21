● Right. The mechanism is "policy → matching → inner SGD with regularizers → final shape → reward", and the policy is gaming the matching-to-final-shape pipe, not the reward. So fixes have to attack one of these links. There are three main attack surfaces, each with very different cost/risk profiles.

  Where the leak is

  state ──► π(c|state) ──► matching c ──► inner SGD (regs) ──► V_final ──► chamfer_sym
                                ▲                 ▲
                                │                 │
                         collapse-friendly   regularizer-dominated:
                         (one-to-many ok)    shape ≈ smoothed(matching pull)

  The leak comes from the second arrow: the inner SGD turns a degenerate matching into a sensible-looking shape, so the policy has no incentive to keep matchings sensible.

  Three places to attack

  1. Constrain the action space (close the collapse channel)

  Force the matching to be a permutation instead of any function S→T. Concretely: replace the per-vertex independent Categorical with sequential sampling without replacement (Plackett-Luce): sample idx_tgt[0] from softmax(S[0, :]); mask that target; sample idx_tgt[1] from softmax over remaining; etc. Closed-form log-prob is sum_k log softmax(S[k, available])
  at sample[k]. Entropy decomposes the same way. Argmax-at-eval becomes a greedy permutation (or full Hungarian if you want optimum, but greedy is fine).

  - Pros: kills collapse by construction. The policy can no longer hack by funneling targets — every target must be used exactly once. Doesn't change the reward, doesn't touch the inner SGD.
  - Cons: changes one file (learned.py + small ripple to FixedMatcher log_prob handling). The reward landscape becomes harder — there's no "easy" win anymore, so REINFORCE has to actually find a good permutation. Variance may go up before it goes down.
  - Effort: medium (1 file, careful with log_prob and entropy). The action space stays the same size order of magnitude (M^M → M!).
  - My take: this is the most principled fix. Recommended first thing to try.

  2. Change what the matching does (make collapse hurt)

  The inner SGD's data term distance_loss averages |V[i] - V_tgt[match_i]|² over pairs — duplicates allowed. Two variants that punish collapse:

  - (a) Per-target coverage term: add to the reward -λ * (M - num_unique_targets(c)) / M. Direct shaping. Easy to implement, but it's a band-aid: REINFORCE will satisfy it minimally rather than learn meaningful matches.
  - (b) Optimal-transport data term in inner SGD: replace distance_loss(matchings) with a Sinkhorn-style soft assignment cost that penalizes mass duplication. This is intrusive — changes the scenario semantics — but it makes the inner SGD itself reject collapsed matchings instead of absorbing them.
  - Pros: doesn't require changing the policy.
  - Cons: (a) is hackish, (b) is a real engineering project.
  - My take: skip unless (1) doesn't work.

  3. Anchor the policy to its initialization (preserve the prior)

  The untrained matcher already has a meaningful inductive bias (random-feature smoothing → roughly local matching). The trained eval reward going from -0.077 → -0.327 means training is destroying a good prior. Standard fix: KL-constraint to the initial policy.

  - Save a frozen copy of the feature extractor at step 0 → π_0.
  - Each batch: compute S_0 with frozen weights, S with current weights; add -β * KL(softmax(S) || softmax(S_0)) to the loss (or use it as a trust region per PPO).
  - Tune β so training improves reward without drifting far from π_0.
  - Pros: directly prevents the observed regression. Doesn't change the protocol.
  - Cons: if the prior is the best policy (untrained > trained on eval), KL-anchoring will just freeze it. That's a useful diagnostic — if the policy can't beat its own prior under any β, the protocol itself isn't learnable in the RL sense and the answer is fix #1 or #2.
  - Effort: low (a few lines). Mostly tuning.
  - My take: worth running as a diagnostic even if not the final fix, because it tells you whether the trained policy can ever beat the random-feature prior given freedom — or whether the protocol is fundamentally broken.

  Honorable mentions

  - Bump entropy_coef from 0.01 → 0.1 or 0.3. Cheap, may delay collapse but won't fix it. Worth pairing with anything else.
  - Reduce regularizer weights in sgd_fixed_match (w_edge, w_laplacian ↓). Forces the matching to determine more of the shape. Risk: instability and self-intersections; the inner SGD becomes a worse optimizer. Mostly diagnostic.
  - Switch protocols entirely: have the policy emit a deformation field directly (skip the matching + inner SGD). This is "different project," not "fix the matcher", but it's the answer if you decide the matching abstraction isn't pulling its weight.

  My recommended order

  1. Diagnostic first: run KL-anchor (#3) with a few β values and see if any improve over the untrained baseline. If none do, the matching protocol is the bottleneck.
  2. If #3 doesn't beat the prior: implement permutation sampling (#1). This is the principled fix.
  3. Only if #1 plateaus: consider #2(b) — Sinkhorn data term in inner SGD.

  Which one do you want to start with? My vote is the KL diagnostic, because it's 10 lines of code and tells you something definitive about whether to invest in #1.
