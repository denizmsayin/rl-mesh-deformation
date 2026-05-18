import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def nearest_point_matching(v_source, v_target):
    """For each source point, pick the nearest target point by Euclidean distance."""
    diffs = v_source[:, None, :] - v_target[None, :, :]
    sq_dist = np.sum(diffs * diffs, axis=2)
    return np.argmin(sq_dist, axis=1)


def evenly_spaced_indices(n, desired_count):
    count = max(1, min(int(round(desired_count)), n))
    return np.unique(np.round(np.linspace(0, n - 1, count)).astype(int))

def draw_edges(ax, vertices, edges, color):
    seg = np.stack((vertices[edges[:, 0]], vertices[edges[:, 1]]), axis=1)
    ax.add_collection(LineCollection(seg, colors=color, linewidths=1.0, alpha=0.55, zorder=1))


def draw_matching_lines(ax, v_s, v_t, match_idx, line_colors, line_width, line_alpha, line_indices):
    seg = np.stack((v_s[line_indices], v_t[match_idx[line_indices]]), axis=1)
    colors = np.array(line_colors[line_indices], dtype=float)
    colors[:, 3] = line_alpha
    ax.add_collection(
        LineCollection(seg, colors=colors, linewidths=max(line_width, 0.2), zorder=2)
    )


def draw_sparse_arrows(ax, v_s, v_t, match_idx, arrow_colors, line_width, line_alpha, arrow_indices):
    arrow_lw = max(line_width * 2.2, 0.75)
    for i in arrow_indices:
        color = np.array(arrow_colors[i], dtype=float)
        color[3] = min(1.0, line_alpha + 0.15)
        ax.annotate(
            "",
            xy=v_t[match_idx[i]],
            xytext=v_s[i],
            arrowprops={
                "arrowstyle": "-|>",
                "color": tuple(color),
                "lw": arrow_lw,
                "shrinkA": 0.0,
                "shrinkB": 2.0,
                "mutation_scale": 10.0,
            },
            zorder=4,
        )


# matching here is a vector that for each point in the start shape, matches one point in the target shape.
# where the index in the array is the index of the start point.
def visualize_matching(
    shape_start,
    shape_target,
    matching=None,
    mode="lines",
    matching_mode="auto",
    max_lines=250,
    show_arrows=False,
    arrow_percentage=0.25,
    point_size=24,
    line_width=0.35,
    line_alpha=0.4,
    ax=None,
):
    """Visualize source/target shapes and source->target matching."""
    if mode not in {"gradient", "lines", "both"}:
        raise ValueError("mode must be one of: 'gradient', 'lines', 'both'.")
    if matching_mode not in {"nearest", "provided", "auto"}:
        raise ValueError("matching_mode must be one of: 'nearest', 'provided', 'auto'.")
    if not (0.0 <= arrow_percentage <= 1.0):
        raise ValueError("arrow_percentage must be in [0, 1].")

    v_s, l_s = shape_start
    v_t, l_t = shape_target

    n_source, n_target = len(v_s), len(v_t)
    use_nearest = matching_mode == "nearest" or (matching_mode == "auto" and matching is None)
    if use_nearest:
        match_idx = nearest_point_matching(v_s, v_t)
    else:
        match_idx = np.asarray(matching, dtype=int)

    created_figure = ax is None
    if created_figure:
        _, ax = plt.subplots(figsize=(7, 7))

    draw_edges(ax, v_s, l_s, "#1f77b4")
    draw_edges(ax, v_t, l_t, "#ff7f0e")

    if mode in {"gradient", "both"}:
        cmap = plt.get_cmap("turbo")
        source_colors = cmap(np.linspace(0.0, 1.0, max(n_source, 2)))[:n_source]
        target_colors = np.full((n_target, 4), (0.72, 0.72, 0.72, 0.82), dtype=float)
        for target_i in range(n_target):
            src_ids = np.where(match_idx == target_i)[0]
            if src_ids.size:
                target_colors[target_i] = source_colors[src_ids].mean(axis=0)
    else:
        source_colors = np.tile(np.array([0.1216, 0.4667, 0.7059, 1.0]), (n_source, 1))
        target_colors = np.tile(np.array([1.0, 0.4980, 0.0549, 1.0]), (n_target, 1))

    ax.scatter(v_s[:, 0], v_s[:, 1], c=source_colors, s=point_size, zorder=3)
    ax.scatter(v_t[:, 0], v_t[:, 1], c=target_colors, s=point_size, zorder=3)

    n_lines_target = min(n_source, max_lines)
    line_indices = evenly_spaced_indices(n_source, n_lines_target)
    draw_matching_lines(ax, v_s, v_t, match_idx, source_colors, line_width, line_alpha, line_indices)

    if show_arrows and len(line_indices) > 0:
        desired_arrows = int(round(len(line_indices) * arrow_percentage))
        arrow_count = max(1, min(int(round(desired_arrows)), len(line_indices)))
        arrow_local = evenly_spaced_indices(len(line_indices), arrow_count)
        arrow_indices = line_indices[arrow_local]
        draw_sparse_arrows(
            ax, v_s, v_t, match_idx, source_colors, line_width, line_alpha, arrow_indices
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)

    if created_figure:
        plt.show()

    return ax, match_idx
