from typing import List

from rlmd.ops import Matching, knn_match


class Knn3dMatcher:
    """Nearest-neighbor matching computed directly in ambient (spatial) coordinates."""

    name = "knn_3d"

    def __init__(self, bidirectional: bool = True):
        self.bidirectional = bidirectional

    def __call__(self, poly_src, poly_tgt) -> List[Matching]:
        P_src, _, n_src, _ = poly_src
        P_tgt, _, n_tgt, _ = poly_tgt
        return knn_match(P_src, n_src, P_tgt, n_tgt, bidirectional=self.bidirectional)
