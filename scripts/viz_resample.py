"""Quick visual sanity check of resample_uniform_polyline.

Runs the resampler on a handful of shapes at a few target counts and saves a
PNG so the result can be eyeballed.

Usage: pixi run python scripts/viz_resample.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from rlmd.batching import pad_polylines
from rlmd.data.generation import Circle, Hexagon, Star, Triangle
from rlmd.ops import resample_uniform_polyline


def _sequential_l(n):
    return np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int64)


def _wonky_blob(seed=0, n=40):
    """Closed random-radius polygon; deliberately irregular."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = 0.7 + 0.4 * rng.random(n)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1).astype(np.float32)


def _shape_inputs():
    """Return list of (label, V) numpy polylines."""
    def pts(shape):
        return np.asarray(shape.points, dtype=np.float32)

    return [
        ("triangle (~30 pts)", pts(Triangle(num_points=30))),
        ("hexagon (~30 pts)", pts(Hexagon(num_points=30))),
        ("star-5 (~50 pts)", pts(Star(num_points=50, n_tips=5, inner_radius=0.45))),
        ("circle (17 pts)", pts(Circle(num_points=17))),
        ("wonky blob (40 pts)", _wonky_blob(seed=2, n=40)),
    ]


def _resample(V_np, M):
    V, L, nv, _ = pad_polylines([V_np], [_sequential_l(V_np.shape[0])])
    V_new, _, _, _ = resample_uniform_polyline(V, L, nv, M)
    return V_new[0].numpy()


def _draw_input(ax, V):
    closed = np.concatenate([V, V[:1]], axis=0)
    ax.plot(closed[:, 0], closed[:, 1], color="tab:gray", lw=1.2, alpha=0.5,
            zorder=1, label=f"input ({V.shape[0]} pts)")
    ax.scatter(V[:, 0], V[:, 1], facecolors="none", edgecolors="tab:gray",
               s=70, lw=1.1, zorder=2)


def _draw_resampled(ax, V):
    closed = np.concatenate([V, V[:1]], axis=0)
    ax.plot(closed[:, 0], closed[:, 1], color="tab:blue", lw=0.8, alpha=0.7,
            zorder=3, label=f"resampled (M={V.shape[0]})")
    ax.scatter(V[:, 0], V[:, 1], color="tab:blue", s=10, zorder=4)
    # Ring the first vertex.
    ax.scatter(V[:1, 0], V[:1, 1], facecolors="none", edgecolors="tab:red",
               s=130, lw=1.8, zorder=5, label="first vertex")


def main():
    M_values = [32, 64, 128]
    shapes = _shape_inputs()

    fig, axes = plt.subplots(len(shapes), len(M_values),
                             figsize=(4.4 * len(M_values), 3.8 * len(shapes)),
                             squeeze=False)

    for row, (label, V0) in enumerate(shapes):
        for col, M in enumerate(M_values):
            ax = axes[row][col]
            V_new = _resample(V0, M)

            _draw_input(ax, V0)
            _draw_resampled(ax, V_new)

            ax.set_aspect("equal")
            ax.set_title(f"{label}  ->  M={M}")
            ax.grid(True, lw=0.3, alpha=0.4)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("resample_uniform_polyline: gray = input vertices (hollow), "
                 "blue = resampled vertices, red ring = first vertex",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "viz_resample.png")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
