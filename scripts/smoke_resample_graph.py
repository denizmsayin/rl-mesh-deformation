"""Visual smoke test for resample_uniform_graph.

Resamples a circle grid and a hexagon grid at several M values and saves a grid
of plots (rows = shape, cols = M) so the arc-length redistribution and junction
pinning can be eyeballed. Standalone (no Hydra).

    pixi run python scripts/smoke_resample_graph.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from rlmd.batching import pad_polylines
from rlmd.data.generation import Grid
from rlmd.ops import resample_uniform_graph

OUT_PATH = "outputs/smoke_resample_graph.png"
CELL_SHAPES = ["circle", "hexagon", "octagon"]
M_VALUES = [16, 64, 256]
_EDGE_COLOR = "#1f77b4"
_JUNCTION_COLOR = "#d62728"


def _junctions(n, edges):
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return np.where(deg != 2)[0]


def _draw(ax, V, edges, junctions=None):
    seg = np.stack((V[edges[:, 0]], V[edges[:, 1]]), axis=1)
    ax.add_collection(LineCollection(seg, colors=_EDGE_COLOR, linewidths=1.0, alpha=0.6))
    ax.scatter(V[:, 0], V[:, 1], s=8, c=_EDGE_COLOR, zorder=3)
    if junctions is not None and len(junctions) > 0:
        ax.scatter(V[junctions, 0], V[junctions, 1], s=40, c=_JUNCTION_COLOR,
                   zorder=4, edgecolors="white", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n_rows, n_cols = len(CELL_SHAPES), len(M_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for r, cell in enumerate(CELL_SHAPES):
        g = Grid(num_points=480, cell_shape=cell, rows=2, cols=2)
        V_np = g.get_points().astype(np.float32)
        L_np = g.get_edges().astype(np.int64)
        V, L, nv, ne = pad_polylines([V_np], [L_np])

        for c, M in enumerate(M_VALUES):
            V_new, L_new, nv_new, ne_new = resample_uniform_graph(V, L, nv, ne, M)
            out_V = V_new[0].numpy()
            out_E = L_new[0, : int(ne_new[0])].numpy()
            jx = _junctions(out_V.shape[0], out_E)
            ax = axes[r][c]
            _draw(ax, out_V, out_E, junctions=jx)
            ax.set_title(f"{cell} grid — M={M}  (V={int(nv_new[0])}, E={int(ne_new[0])})",
                         fontsize=10)

    fig.suptitle("resample_uniform_graph: arc-length resampling (red = pinned junctions)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
