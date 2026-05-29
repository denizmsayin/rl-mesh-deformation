import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv


def _padded_to_pyg(V, L, num_verts, num_edges):
    """Convert a padded batch to a flat PyG-style graph.

    Padded vertices/edges (tail positions beyond num_verts / num_edges) are
    dropped. Valid vertex i of batch item b maps to global index
    offset[b] + i, where offset is the exclusive cumsum of num_verts; this is
    only valid because padding is at the tail (arange < num_verts).

    Edges are made undirected (both directions added) and self-loops are
    appended so every node sees its own feature under max aggregation.

    Returns:
        x: (sumN, 2) float node features (coords).
        edge_index: (2, E) long.
        node_mask: (B, N_max) bool — where valid nodes sit in the padded layout.
    """
    B, N_max, _ = V.shape
    M_max = L.shape[1]
    device = V.device

    node_mask = torch.arange(N_max, device=device)[None, :] < num_verts[:, None]
    x = V[node_mask]                                            # (sumN, 2)

    offset = torch.zeros(B, dtype=torch.long, device=device)
    offset[1:] = torch.cumsum(num_verts, dim=0)[:-1]           # exclusive cumsum

    edge_mask = torch.arange(M_max, device=device)[None, :] < num_edges[:, None]
    a = (L[..., 0] + offset[:, None])[edge_mask]               # (E0,) global src
    b = (L[..., 1] + offset[:, None])[edge_mask]               # (E0,) global dst

    self_idx = torch.arange(x.shape[0], device=device)
    src = torch.cat([a, b, self_idx])
    dst = torch.cat([b, a, self_idx])
    edge_index = torch.stack([src, dst], dim=0)               # (2, E)
    return x, edge_index, node_mask


class PolygonGNN(nn.Module):
    """EdgeConv (DGCNN operator) feature extractor over arbitrary polyline graphs.

    Unlike PolygonCNN, this makes no single-closed-cycle assumption: it operates
    on the real (V, L) graph, so welded / multi-loop shapes such as grids work
    without resampling. Each layer is EdgeConv:

        h_i' = max_{j in N(i)} MLP(h_i ‖ h_j − h_i)

    where the neighbourhood comes from L (treated as undirected, with self-loops
    added). On the first layer h = V, so h_j − h_i is the relative coordinate.

    Args mirror PolygonCNN where sensible so configs/usage stay parallel.

    forward(V, L, num_verts, num_edges=None) -> (B, N_max, out_channels),
    zeroed at padded vertex positions. num_edges may be omitted, in which case
    it is derived from L (rows with any negative index are padding).
    """

    def __init__(self, in_channels=2, hidden_channels=(64, 64, 128),
                 out_channels=128, aggr="max", layernorm=True, residual=True):
        super().__init__()
        dims = [in_channels, *hidden_channels, out_channels]
        # DGCNN-style EdgeConv with a 2-layer message MLP per edge:
        # MLP(h_i ‖ h_j − h_i) = Linear -> ReLU -> Linear. A single Linear (the
        # earlier version) is too weak to build discriminative per-vertex
        # embeddings on near-homogeneous shapes (e.g. a uniformly resampled
        # circle), leaving the matcher's softmax near-uniform.
        self.convs = nn.ModuleList([
            EdgeConv(
                nn.Sequential(
                    nn.Linear(2 * dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.Linear(dims[i + 1], dims[i + 1]),
                ),
                aggr=aggr,
            )
            for i in range(len(dims) - 1)
        ])
        if layernorm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 2)
            ])
        else:
            self.norms = None
        self.residual = residual

    def forward(self, V, L, num_verts, num_edges=None):
        if num_edges is None:
            num_edges = (L >= 0).all(dim=-1).sum(dim=-1).long()

        x, edge_index, node_mask = _padded_to_pyg(V, L, num_verts, num_edges)

        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            if i < last:
                if self.norms is not None:
                    h = self.norms[i](h)
                h = F.relu(h)
            # Residual when widths match (eases optimization / depth).
            x = x + h if (self.residual and h.shape[-1] == x.shape[-1]) else h

        B, N_max, _ = V.shape
        out = V.new_zeros(B, N_max, x.shape[-1])
        out[node_mask] = x
        return out
