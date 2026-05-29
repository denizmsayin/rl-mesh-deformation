import warnings
from collections import namedtuple

import numpy as np
import torch

from rlmd.models.polygon_cnn import check_sequential_l


def resample_uniform_polyline(V, L, num_verts, M, check_l=False):
    """
    Resample each closed polyline to M uniformly arc-length-spaced vertices.

    The first output vertex always coincides with V[..., 0, :] (s = 0), so the
    canonical orientation of the input polyline is preserved.

    Assumes L encodes the canonical sequential ordering
    ``L[b, i] == (i, (i+1) mod num_verts[b])`` for valid i. Set check_l=True to
    validate.

    Differentiable in V via gather + linear interpolation.

    Args:
        V: (B, N_max, D) float — padded vertex coords.
        L: (B, M_max, 2) long — edges; only consulted when check_l=True.
        num_verts: (B,) long — valid vertices per item.
        M: int — output vertex count.
        check_l: if True, validate sequential ordering.

    Returns:
        V_new:  (B, M, D) float — resampled vertices.
        L_new:  (B, M, 2) long — canonical sequential edges.
        nv_new: (B,) long — all equal to M (vertex count).
        ne_new: (B,) long — all equal to M (edge count; closed cycle).
    """
    if check_l:
        check_sequential_l(L, num_verts)

    B, N_max, D = V.shape
    device = V.device
    dtype = V.dtype

    # Index of the "next" vertex for each position, cyclic within each item's
    # valid run of length n. Padded positions don't matter; we set them to 0.
    idx = torch.arange(N_max, device=device)
    idx_b = idx[None, :].expand(B, -1)                           # (B, N_max)
    n = num_verts[:, None]                                       # (B, 1)
    valid = idx_b < n                                             # (B, N_max)
    n_safe = n.clamp(min=1)
    next_idx = torch.where(valid, (idx_b + 1) % n_safe, torch.zeros_like(idx_b))

    next_idx_d = next_idx.unsqueeze(-1).expand(-1, -1, D)        # (B, N_max, D)
    V_next = torch.gather(V, 1, next_idx_d)                      # (B, N_max, D)

    # Per-edge length, zeroed at padded positions.
    seg = torch.linalg.vector_norm(V_next - V, dim=-1)           # (B, N_max)
    seg = seg * valid.to(seg.dtype)

    # Cumulative arc length, padded with a leading zero so that
    # cum[b, i] = sum_{j<i} seg[b, j] and cum[b, n] = total perimeter.
    zero_col = torch.zeros(B, 1, device=device, dtype=seg.dtype)
    cum = torch.cat([zero_col, seg.cumsum(dim=1)], dim=1)        # (B, N_max + 1)

    P = cum.gather(1, n)                                         # (B, 1) total perimeter

    # Query arc-length positions s_k = k/M * P, k = 0..M-1 (k = M would duplicate s = 0).
    k = torch.arange(M, device=device, dtype=dtype)
    s = (k / M)[None, :] * P                                     # (B, M)

    # Find edge index e such that cum[e] <= s < cum[e+1]. searchsorted with
    # right=True returns the first index where cum > s, in [1, n]; subtracting 1
    # lands in [0, n-1].
    e = torch.searchsorted(cum, s, right=True) - 1               # (B, M)
    e = e.clamp(min=0, max=N_max - 1)

    e_d = e.unsqueeze(-1).expand(-1, -1, D)                      # (B, M, D)
    V_e = torch.gather(V, 1, e_d)                                # (B, M, D)
    V_e_next = torch.gather(V_next, 1, e_d)                      # (B, M, D)
    seg_e = seg.gather(1, e)                                     # (B, M)
    cum_e = cum.gather(1, e)                                     # (B, M)

    t = (s - cum_e) / seg_e.clamp(min=torch.finfo(seg.dtype).tiny)  # (B, M)
    V_new = V_e + t.unsqueeze(-1) * (V_e_next - V_e)             # (B, M, D)

    i_out = torch.arange(M, device=device)
    L_new = torch.stack([i_out, (i_out + 1) % M], dim=-1)
    L_new = L_new[None, :, :].expand(B, -1, -1).contiguous()
    nv_new = torch.full((B,), M, dtype=torch.long, device=device)
    ne_new = torch.full((B,), M, dtype=torch.long, device=device)

    return V_new, L_new, nv_new, ne_new


# ---------------------------------------------------------------------------
# Generalized arc-length resampling for arbitrary polyline graphs.
#
# A general shape is a 1-complex (union of edges). The only topologically
# meaningful vertices are junctions (degree != 2: branch points such as grid
# welds, and open endpoints). These are *anchors* kept exactly; the degree-2
# runs between them ("chains") are resampled uniformly by arc length, exactly
# like the single-cycle case. A pure closed cycle has zero junctions, so it is
# one loop chain and reduces to resample_uniform_polyline (handled by an early
# delegation, so the cyclic output is bit-for-bit the original).
# ---------------------------------------------------------------------------

_GraphPlan = namedtuple("_GraphPlan", ["gather_a", "gather_b", "t", "L_out"])
_PLAN_CACHE = {}


def _trace_chains(n, edges):
    """Decompose a simple undirected graph into chains.

    Returns (chains, junctions) where junctions is the sorted list of vertices
    with degree != 2 and chains is a list of (ordered_vertices, is_loop):
    path chains run between two junctions (endpoints inclusive); loop chains are
    anchor-free cycles (vertices listed once, closing edge implicit).
    """
    adj = [[] for _ in range(n)]
    for eid, (a, b) in enumerate(edges):
        a, b = int(a), int(b)
        adj[a].append((b, eid))
        adj[b].append((a, eid))
    deg = [len(a) for a in adj]
    junctions = [v for v in range(n) if deg[v] != 2]
    used = [False] * len(edges)
    chains = []

    def _step(cur, came):
        for nb, e in adj[cur]:
            if e != came:
                return nb, e
        return None, None

    # Path chains: walk each unused edge out of a junction until the next junction.
    for j in junctions:
        for nbr, eid in adj[j]:
            if used[eid]:
                continue
            used[eid] = True
            chain = [j, nbr]
            cur, came = nbr, eid
            while deg[cur] == 2:
                nxt, ne = _step(cur, came)
                used[ne] = True
                chain.append(nxt)
                cur, came = nxt, ne
            chains.append((chain, False))

    # Loop chains: remaining unused edges form anchor-free cycles.
    for eid, (a, b) in enumerate(edges):
        if used[eid]:
            continue
        a, b = int(a), int(b)
        used[eid] = True
        chain = [a]
        cur, came = b, eid
        while cur != a:
            chain.append(cur)
            nxt, ne = _step(cur, came)
            used[ne] = True
            cur, came = nxt, ne
        chains.append((chain, True))

    return chains, junctions


def _allocate_segments(chains, V0, M, n_junctions):
    """Largest-remainder allocation of M - n_junctions new vertices to chains,
    proportional to chain arc length. Returns a list of output segment counts.

    A path chain with s segments contributes s-1 new vertices (its 2 endpoints
    are shared junctions); a loop chain with s segments contributes s. Floors:
    path s>=1 (connectivity), loop s>=3 (a valid loop).
    """
    lens = []
    for verts, is_loop in chains:
        pv = V0[verts]
        if is_loop:
            d = np.linalg.norm(np.roll(pv, -1, axis=0) - pv, axis=1).sum()
        else:
            d = np.linalg.norm(pv[1:] - pv[:-1], axis=1).sum()
        lens.append(float(d))
    lens = np.asarray(lens)
    floors = np.array([3 if is_loop else 0 for _, is_loop in chains])

    target = M - n_junctions
    if target < int(floors.sum()):
        raise ValueError(
            f"resample_uniform_graph: M={M} too small for this topology; "
            f"needs at least {n_junctions + int(floors.sum())} vertices "
            f"({n_junctions} junctions + {int(floors.sum())} loop minimums)."
        )

    base = target - int(floors.sum())
    w = lens / lens.sum() if lens.sum() > 0 else np.full(len(lens), 1.0 / len(lens))
    raw = w * base
    add = np.floor(raw).astype(int)
    rem = int(base - add.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    for i in range(rem):
        add[order[i]] += 1
    new = floors + add
    return [int(nc) if is_loop else int(nc + 1)
            for nc, (_, is_loop) in zip(new, chains)]


def _chain_outputs(verts, is_loop, seg, V0):
    """Per-output-vertex (in_a, in_b, t) referencing input vertex indices.

    Path chains return seg+1 entries (both junction endpoints as identity
    t=0); loop chains return seg entries around the cycle starting at verts[0].
    """
    tiny = float(np.finfo(np.float64).tiny)
    pv = V0[verts]
    if is_loop:
        seglen = np.linalg.norm(np.roll(pv, -1, axis=0) - pv, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        P = cum[-1]
        out = []
        for k in range(seg):
            sk = k / seg * P
            i = min(max(int(np.searchsorted(cum, sk, side="right")) - 1, 0), len(verts) - 1)
            t = (sk - cum[i]) / max(seglen[i], tiny)
            out.append((int(verts[i]), int(verts[(i + 1) % len(verts)]), float(t)))
        return out

    seglen = np.linalg.norm(pv[1:] - pv[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    P = cum[-1]
    out = []
    for k in range(seg + 1):
        if k == 0:
            out.append((int(verts[0]), int(verts[0]), 0.0))
        elif k == seg:
            out.append((int(verts[-1]), int(verts[-1]), 0.0))
        else:
            sk = k / seg * P
            i = min(max(int(np.searchsorted(cum, sk, side="right")) - 1, 0), len(verts) - 2)
            t = (sk - cum[i]) / max(seglen[i], tiny)
            out.append((int(verts[i]), int(verts[i + 1]), float(t)))
    return out


def _build_graph_plan(V0, edges, M):
    """Build the (topology + arc-length-ratio dependent) resampling plan.

    Junctions occupy output indices 0..J-1; chain interiors/loops follow. Emits
    a warning if any chain resamples to fewer than 3 segments.
    """
    chains, junctions = _trace_chains(V0.shape[0], edges)
    segs = _allocate_segments(chains, V0, M, len(junctions))

    gather_a, gather_b, tlist = [], [], []
    jmap = {}
    for k, jv in enumerate(junctions):
        jmap[jv] = k
        gather_a.append(jv)
        gather_b.append(jv)
        tlist.append(0.0)
    next_idx = len(junctions)
    edges_out = []
    n_short = 0

    for (verts, is_loop), seg in zip(chains, segs):
        if seg < 3:
            n_short += 1
        outs = _chain_outputs(verts, is_loop, seg, V0)
        gidx = []
        if is_loop:
            for a, b, t in outs:
                gidx.append(next_idx)
                gather_a.append(a)
                gather_b.append(b)
                tlist.append(t)
                next_idx += 1
            for i in range(len(gidx)):
                edges_out.append((gidx[i], gidx[(i + 1) % len(gidx)]))
        else:
            for pos, (a, b, t) in enumerate(outs):
                if pos == 0:
                    gidx.append(jmap[int(verts[0])])
                elif pos == len(outs) - 1:
                    gidx.append(jmap[int(verts[-1])])
                else:
                    gidx.append(next_idx)
                    gather_a.append(a)
                    gather_b.append(b)
                    tlist.append(t)
                    next_idx += 1
            for i in range(len(gidx) - 1):
                edges_out.append((gidx[i], gidx[i + 1]))

    if next_idx != M:
        raise AssertionError(
            f"resample_uniform_graph plan produced {next_idx} vertices, expected {M}.")
    if n_short > 0:
        warnings.warn(
            f"resample_uniform_graph: {n_short} chain(s) resampled to fewer than "
            f"3 segments at M={M}; short curved chains may be flattened. "
            f"Increase M for higher fidelity.",
            stacklevel=3,
        )

    return _GraphPlan(
        gather_a=np.asarray(gather_a, dtype=np.int64),
        gather_b=np.asarray(gather_b, dtype=np.int64),
        t=np.asarray(tlist, dtype=np.float64),
        L_out=np.asarray(edges_out, dtype=np.int64),
    )


def _is_pure_cycle(L, num_verts, num_edges):
    """True iff every valid vertex in every batch item has degree exactly 2."""
    M_max = L.shape[1]
    N_max = int(num_verts.max().item()) if num_verts.numel() else 0
    device = L.device
    e_valid = (torch.arange(M_max, device=device)[None, :] < num_edges[:, None])
    a = L[..., 0].clamp(min=0)
    b = L[..., 1].clamp(min=0)
    n_cols = max(N_max, int(a.max().item()) + 1 if a.numel() else 1)
    deg = torch.zeros(L.shape[0], n_cols, device=device)
    ones = e_valid.to(deg.dtype)
    deg.scatter_add_(1, a, ones)
    deg.scatter_add_(1, b, ones)
    v_valid = (torch.arange(n_cols, device=device)[None, :] < num_verts[:, None])
    return bool(((deg == 2) | ~v_valid).all().item())


def resample_uniform_graph(V, L, num_verts, num_edges, M):
    """Resample an arbitrary polyline graph to M arc-length-uniform vertices.

    Generalizes resample_uniform_polyline to multi-loop / welded graphs (e.g.
    grids). Junctions (degree != 2) are kept exactly as anchors; the degree-2
    chains between them are resampled uniformly by arc length, with the M - J
    free vertices distributed across chains proportional to length.

    Pure-cycle batches (every vertex degree 2) delegate to
    resample_uniform_polyline, so that case is identical to before (and still
    supports per-item variable length). Graph batches require a homogeneous
    topology across the batch (same num_verts, num_edges, and edge list), which
    holds for the homogeneous batches the pipeline produces.

    Differentiable in V (gather + linear interpolation). Returns the same
    4-tuple shape as resample_uniform_polyline: (V_new (B,M,D), L_new (B,E,2),
    nv_new (B,), ne_new (B,)).
    """
    if _is_pure_cycle(L, num_verts, num_edges):
        return resample_uniform_polyline(V, L, num_verts, M)

    B = V.shape[0]
    device = V.device
    n0 = int(num_verts[0].item())
    e0 = int(num_edges[0].item())
    if not (bool((num_verts == n0).all()) and bool((num_edges == e0).all())):
        raise ValueError(
            "resample_uniform_graph: graph batches must be topologically "
            "homogeneous (equal num_verts / num_edges across items).")
    if not bool((L[:, :e0] == L[0:1, :e0]).all()):
        raise ValueError(
            "resample_uniform_graph: graph batches must share an identical edge "
            "list across items.")

    V0 = V[0, :n0].detach().cpu().numpy().astype(np.float64)
    edges0 = L[0, :e0].cpu().numpy()
    key = (M, n0, e0, edges0.tobytes())
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_graph_plan(V0, edges0, M)
        _PLAN_CACHE[key] = plan

    ga = torch.as_tensor(plan.gather_a, device=device)
    gb = torch.as_tensor(plan.gather_b, device=device)
    t = torch.as_tensor(plan.t, device=device, dtype=V.dtype)
    Va = V.index_select(1, ga)
    Vb = V.index_select(1, gb)
    V_new = Va + t[None, :, None] * (Vb - Va)

    L_out = torch.as_tensor(plan.L_out, device=device, dtype=torch.long)
    L_new = L_out[None].expand(B, -1, -1).contiguous()
    nv_new = torch.full((B,), M, dtype=torch.long, device=device)
    ne_new = torch.full((B,), L_out.shape[0], dtype=torch.long, device=device)
    return V_new, L_new, nv_new, ne_new
