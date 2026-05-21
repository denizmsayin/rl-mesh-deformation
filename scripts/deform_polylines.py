import numpy as np
import torch
from tqdm import tqdm

from rlmd.data.generation import ShapeGenerator
from rlmd.batching import pad_polylines
from rlmd.ops import (
    distance_loss,
    knn_match,
    polyline_edge_loss,
    polyline_laplacian_smoothing,
    polyline_normal_consistency,
    sample_points_from_polylines,
)
from rlmd.visualization.visualize import plot_polylines_initial_vs_final


BATCH_SIZE = 10
NUM_ITERS = 2000
NUM_POINTS = 60
NUM_SAMPLES = 500
TARGET_SHAPE = 'star'
TRANSLATION_RADIUS = 5
LR = 0.1   # per-item; loss ops sum over batch, so lr is batch-size-invariant
MOMENTUM = 0.9

W_CHAMFER = 1.0
W_EDGE = 1.0
W_NORMAL = 0.01
W_LAPLACIAN = 0.1


def build_batch():
    gen = ShapeGenerator()
    base_src = gen.get_base_shape('circle', num_points=NUM_POINTS)
    base_tgt = gen.get_base_shape(TARGET_SHAPE, num_points=NUM_POINTS)

    angles = np.linspace(0.0, 2 * np.pi, BATCH_SIZE, endpoint=False)
    n_still = BATCH_SIZE // 2
    t_angles = np.linspace(0.0, 2 * np.pi, BATCH_SIZE - n_still, endpoint=False)
    centers = [(0.0, 0.0)] * n_still + [
        (TRANSLATION_RADIUS * np.cos(a), TRANSLATION_RADIUS * np.sin(a)) for a in t_angles
    ]

    src_pts = [base_src.get_points() for _ in range(BATCH_SIZE)]
    src_edges = [base_src.get_edges() for _ in range(BATCH_SIZE)]

    tgt_pts = []
    for a, c in zip(angles, centers):
        cos_a, sin_a = np.cos(a), np.sin(a)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        tgt_pts.append(base_tgt.get_points() @ R.T + np.array(c))
    tgt_edges = [base_tgt.get_edges() for _ in range(BATCH_SIZE)]

    V_src, L_src, nv_src = pad_polylines(src_pts, src_edges)
    V_tgt, L_tgt, nv_tgt = pad_polylines(tgt_pts, tgt_edges)
    return V_src, L_src, nv_src, V_tgt, L_tgt, nv_tgt


def compute_losses(V, L, nv, V_tgt, L_tgt, nv_tgt):
    P = sample_points_from_polylines(V, L, nv, NUM_SAMPLES)
    P_tgt = sample_points_from_polylines(V_tgt, L_tgt, nv_tgt, NUM_SAMPLES)
    n_samples = torch.full((V.shape[0],), NUM_SAMPLES, dtype=torch.long)
    matchings = knn_match(P, n_samples, P_tgt, n_samples, bidirectional=True)
    l_chamfer = distance_loss(P, P_tgt, matchings, p=2)
    l_edge = polyline_edge_loss(V, L, nv)
    l_normal = polyline_normal_consistency(V, L, nv)
    l_laplacian = polyline_laplacian_smoothing(V, L, nv)
    total = (W_CHAMFER * l_chamfer + W_EDGE * l_edge
             + W_NORMAL * l_normal + W_LAPLACIAN * l_laplacian)
    return total, l_chamfer, l_edge, l_normal, l_laplacian


def main():
    V_src0, L_src, nv_src, V_tgt, L_tgt, nv_tgt = build_batch()

    deform = torch.zeros_like(V_src0, requires_grad=True)
    optimizer = torch.optim.SGD([deform], lr=LR, momentum=MOMENTUM)

    pbar = tqdm(range(NUM_ITERS))
    for _ in pbar:
        optimizer.zero_grad()
        V = V_src0 + deform
        total, lc, le, ln, ll = compute_losses(V, L_src, nv_src, V_tgt, L_tgt, nv_tgt)
        total.backward()
        optimizer.step()
        pbar.set_postfix(
            total=f'{total.item():.4f}',
            chamfer=f'{lc.item():.4f}',
            edge=f'{le.item():.4f}',
            normal=f'{ln.item():.4f}',
            lap=f'{ll.item():.4f}',
        )

    V_final = (V_src0 + deform).detach()

    print('\nPer-problem final loss:')
    with torch.no_grad():
        for i in range(BATCH_SIZE):
            total, lc, *_ = compute_losses(
                V_final[i:i+1], L_src[i:i+1], nv_src[i:i+1],
                V_tgt[i:i+1], L_tgt[i:i+1], nv_tgt[i:i+1],
            )
            print(f'  problem {i}: total={total.item():.4f} chamfer={lc.item():.4f}')

    plot_polylines_initial_vs_final(
        V_src0, V_final, L_src, nv_src, V_tgt, L_tgt, nv_tgt,
        'deform_polylines.png',
        title_prefix='problem',
    )
    print('saved deform_polylines.png')


if __name__ == '__main__':
    main()
