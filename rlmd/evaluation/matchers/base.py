from typing import List, Protocol, Tuple

import torch

from rlmd.ops import Matching

# A padded polyline graph: (V, L, num_verts, num_edges). Matchers that ignore
# topology (e.g. KNN) only read V and num_verts; learned/graph matchers consume
# L and num_edges too. For the sampled-point path (SgdScenario), L/num_edges may
# be None since sampled points carry no graph.
Polyline = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Matcher(Protocol):
    """
    Establishes correspondences between two padded polyline graphs.

    Implementations may produce one or more Matchings per call (e.g. forward,
    bidirectional). The downstream loss averages over the returned matchings.
    """
    name: str

    def __call__(self, poly_src: Polyline, poly_tgt: Polyline) -> List[Matching]:
        ...
