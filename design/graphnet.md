# Graph network for per-vertex features

Working notes on replacing `PolygonCNN` with a graph neural network (GNN) as the
matcher's feature extractor. The CNN assumes a single closed cycle (circular 1D
convolution over sequentially-ordered vertices); that assumption breaks the
moment a shape is more than one loop — welded grids being the motivating case.
A GNN consumes the `(V, L)` graph directly and makes no such assumption.

This is a discussion document, not a spec. The open decision is which
message-passing backbone to build first; see the bottom.

## Context recap

- A "shape" is a pair of tensors: `V (B, N, 2)` vertex coords and `L (B, M, 2)`
  edge index pairs, plus `num_verts (B,)` and `num_edges (B,)` length tensors
  (batches are zero-padded; `L` padded with `-1`).
- The matcher's feature extractor maps `(V, L, num_verts[, num_edges]) ->
  (B, N_max, D)` per-vertex features, zeroed at padded positions. Source and
  target features are compared by dot product to produce correspondence logits
  (`rlmd/evaluation/matchers/learned.py`).
- `PolygonCNN` (`rlmd/models/polygon_cnn.py`) satisfies that contract but only
  for a single closed polygon: it circular-pads each item's valid vertex run and
  convolves, ignoring `L` except for an optional sequentiality check. Resampling
  (`resample_uniform_polyline`) is what forces every shape into that canonical
  cyclic form — and it is exactly what garbles a grid (verified: resampled grid
  vertices land up to ~0.79 away from any true grid vertex).
- PyG (`torch_geometric`) is now installed, so we build on it rather than
  hand-rolling scatter/message-passing.

## The mental model

Every PyG layer is one abstraction, `MessagePassing`, with three steps:

1. **message** — for each edge `j -> i`, compute a vector from the neighbour's
   features `x_j` (and optionally the receiver `x_i` and an edge attribute
   `e_ji`).
2. **aggregate** — combine all incoming messages at node `i` with a permutation-
   invariant reducer (sum / mean / max).
3. **update** — combine the aggregate with node `i`'s own features to produce
   the new `x_i`.

"Which GNN" is just "what fills those three slots." Everything below is a
different filling of the same skeleton. Because aggregation is permutation-
invariant over neighbours and applied per-node, the whole stack is permutation-
**equivariant** over nodes — relabel the vertices and the outputs come back in
the same relabelled order. That is the property the CNN faked with sequential
ordering and that we now get for free, on arbitrary topology.

## The menu (and how each fits this problem)

The problem is specific: **small 2D geometric graphs** (polylines, welded
grids) with tiny node degree (≈2–4), where the signal is mostly **coordinates
and relative positions**, and we want **per-node embeddings** whose dot products
give good source→target correspondences.

| Family | message / aggregation | Verdict here |
|---|---|---|
| **GCNConv** (the basic one) | degree-normalised mean of linearly-transformed neighbours; no edge info | Isotropic, can't use geometry. Weak baseline. |
| **SAGEConv** | concat(self, mean/max of neighbours) | Simple, solid baseline. Ignores edge geometry unless coords are baked into node features. |
| **GATv2Conv** | attention-weighted neighbour sum | Lets a node weight neighbours unequally. Less valuable when degree is tiny. |
| **GINConv / GINEConv** | **sum** agg + MLP; GINE also folds in edge features | Most *structurally* expressive (Weisfeiler–Lehman bound). GINE is a clean way to inject relative-position edge features. |
| **EdgeConv / DGCNN** | `MLP(concat(x_i, x_j − x_i))`, **max** agg | Built for point clouds. Uses relative geometry by construction. Strong, simple default for exactly this shape. |
| **NNConv / TransformerConv** | edge-conditioned / attention-weighted messages | Flexible, heavier. Worth it only if edge features get rich. |
| **EGNN / equivariant nets** | jointly update coords + features in an E(n)-equivariant way | Principled: outputs transform correctly under rotation/translation. Most moving parts. |

### Notes per family

- **GCNConv** — the spectral-flavoured layer most people meet first (and what was
  used via pytorch3d before). Normalises by `1/sqrt(deg_i · deg_j)`, sums, applies
  one linear map. No way to pass edge features, no notion of relative position;
  every neighbour is treated identically up to the degree weight. Fine for
  node-classification-on-citation-graphs, underpowered for geometry.
- **SAGEConv** — "sample and aggregate." Keeps the receiver's own features
  separate from the neighbour aggregate (concat then linear), which makes it
  noticeably stronger than GCN in practice. Mean or max pooling. Still geometry-
  blind unless you put coordinates in the node features.
- **GATv2Conv** — computes an attention coefficient per edge and takes a weighted
  sum. `v2` fixes the "static attention" flaw in the original GAT. Useful when
  some neighbours matter more than others, but with degree 2–4 there's little for
  attention to do.
- **GIN / GINE** — sum aggregation plus an MLP update is provably as
  discriminative as the 1-WL test, i.e. maximally expressive among message-
  passing nets at telling graph structures apart. `GINEConv` extends it to accept
  edge attributes, which is the hook for relative positions. Sum aggregation is
  sensitive to degree, which is fine here because degree encodes real structure
  (a welded grid junction has higher degree than a loop vertex).
- **EdgeConv (DGCNN)** — message is `MLP(x_i ‖ x_j − x_i)` with max aggregation.
  The `x_j − x_i` term means relative geometry is used by construction, and on
  the first layer (where `x = V`) that term is literally the relative coordinate.
  This is the standard strong baseline for point-cloud/geometric graphs and is a
  one-liner in PyG.
- **NNConv / TransformerConv** — let an edge MLP generate the weight matrix
  (NNConv) or modulate attention (TransformerConv). More capacity, more params,
  more tuning; reach for these only once simple edge features prove insufficient.
- **EGNN and friends** — maintain separate coordinate and feature channels and
  update coordinates only through equivariant operations, so a rotation/
  translation of the input produces the same rotation/translation of the output.
  The most principled answer to pose, at the cost of a more constrained, more
  complex layer.

## The decision that actually matters: how coordinates enter

This dominates the layer choice. Three options, increasing in sophistication:

- **Absolute coords as node features** — what `PolygonCNN` does today. Pose-
  *aware* but not translation/rotation invariant, so the network must learn to
  undo the augmentation (translate + rotate + isotropic scale).
- **Relative coords as edge features** (`x_j − x_i`) — translation-invariant by
  construction. EdgeConv and GINE both consume this naturally. Good middle
  ground: cheap, and removes the easiest nuisance variable (absolute position).
- **Full E(n)-equivariance** (EGNN) — invariant to rotation + translation as
  well. This is conceptually attractive for *matching*: the matcher compares two
  shapes at independent poses, so per-node embeddings that depend only on
  intrinsic shape "role" (not pose) would make corresponding points align
  regardless of orientation. The costs: more complex layers, and isotropic scale
  is still not handled for free (would need an explicit normalisation, e.g.
  centre-and-rescale by a size statistic before the net).

A useful framing: each step up the list removes a nuisance transformation the
network would otherwise have to learn to ignore, in exchange for implementation
complexity. With heavy rotation augmentation in the data, there is real value in
moving past absolute coordinates.

## Recommendation

Start with **EdgeConv (DGCNN-style)** as the backbone:
`h_i' = max_j MLP(h_i ‖ h_j − h_i)`, a few layers, LayerNorm + ReLU, raw XY as
the initial node feature.

Why first:

- Standard strong baseline for geometric point graphs.
- Uses relative position for free (and exactly the relative coordinate on layer 1).
- Keeps the `(V, L)` graph intact — no cyclic / resample assumption — so welded
  grids work without special handling.
- One PyG layer (`torch_geometric.nn.EdgeConv`), minimal surface area.

Main tradeoff: EdgeConv is **invariant to nothing** by default — absolute
position leaks in through the `h_i` term. If pose-robustness turns out to matter
for matching quality, the clean upgrade path is:

EdgeConv (absolute) → GINE (relative-only messages) → EGNN (full E(n)
equivariance).

## Integration notes (out of scope for the first cut, but constraining)

- The new layer must satisfy the existing feature-extractor contract:
  `forward(V, L, num_verts[, num_edges]) -> (B, N_max, D)`, padded positions
  zeroed, so it slots into `learned.py` where `PolygonCNN` sits today.
- Internally a PyG model wants `edge_index (2, E)` and a flat node tensor with a
  `batch` vector, not our padded `(B, N, 2)` / `(B, M, 2)` layout. The adapter
  (dense-padded ↔ PyG `Batch`) lives at the model boundary; the rest of the
  pipeline keeps speaking the padded-tensor API.
- `_compute_scores` in `learned.py` currently *rebuilds a canonical cyclic `L`*
  from `num_verts` and ignores the real edges — and the training/eval path
  resamples every shape to an `M`-vertex cycle first. For the GNN to see true
  grid topology, that resample-and-recanonicalise path has to be bypassed for
  graph shapes. This is the larger follow-up after the backbone exists.

## Open decision

- **Backbone for the first cut.** Recommendation: EdgeConv. Alternatives on the
  table: GINEConv (relative-only, translation-invariant) or EGNN (full
  equivariance) if we decide pose-invariance is worth the complexity up front.
