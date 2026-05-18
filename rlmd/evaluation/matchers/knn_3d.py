from typing import List

import torch

from rlmd.ops import Matching, knn_match


class Knn3dMatcher:
    """Nearest-neighbor matching computed directly in ambient (spatial) coordinates."""

    name = "knn_3d"

    def __init__(self, bidirectional: bool = True):
        self.bidirectional = bidirectional

    def __call__(
        self,
        P_src: torch.Tensor,
        n_src: torch.Tensor,
        P_tgt: torch.Tensor,
        n_tgt: torch.Tensor,
    ) -> List[Matching]:
        return knn_match(P_src, n_src, P_tgt, n_tgt, bidirectional=self.bidirectional)
