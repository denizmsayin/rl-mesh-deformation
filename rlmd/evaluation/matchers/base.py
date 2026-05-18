from typing import List, Protocol

import torch

from rlmd.ops import Matching


class Matcher(Protocol):
    """
    Establishes correspondences between two padded point clouds.

    Implementations may produce one or more Matchings per call (e.g. forward,
    bidirectional). The downstream loss averages over the returned matchings.
    """
    name: str

    def __call__(
        self,
        P_src: torch.Tensor,
        n_src: torch.Tensor,
        P_tgt: torch.Tensor,
        n_tgt: torch.Tensor,
    ) -> List[Matching]:
        ...
