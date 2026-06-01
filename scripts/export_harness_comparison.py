"""Export initial and final deformation PDFs for comparing two harness configurations.

Re-runs each configured scenario on the first ``num_samples`` eval pairs and writes
fixed-size publication panels (same styling as ``export_deformation_stages.py``).

Layout under ``output_dir`` (default ``fig/`` at repo root)::

    init/{idx:03d}.pdf       # src + tgt before deformation (shared)
    chamfer/{idx:03d}.pdf    # learned fixed-match final
    knn/{idx:03d}.pdf        # KNN baseline final

Example (matches the two latest evaluate_harness runs on 2026-06-01)::

  pixi run python scripts/export_harness_comparison.py
"""

import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from rlmd.dataset import shape_collate_fn
from rlmd.evaluation.harness import _build_dataset, _resolve_device
from rlmd.ops import resample_uniform_polyline
from rlmd.visualization.visualize import save_src_tgt_pair_cells


def _load_eval_batch(cfg: DictConfig, num_samples: int, device: torch.device):
    dataset_src = _build_dataset(cfg.dataset_src, cfg.eval_num_samples, cfg.seed_src)
    dataset_tgt = _build_dataset(cfg.dataset_tgt, cfg.eval_num_samples, cfg.seed_tgt)
    if num_samples > len(dataset_src) or num_samples > len(dataset_tgt):
        raise ValueError(
            f"num_samples={num_samples} exceeds eval subset size {len(dataset_src)}."
        )

    items_src = [dataset_src[i] for i in range(num_samples)]
    items_tgt = [dataset_tgt[i] for i in range(num_samples)]
    V_src, L_src, nv_src, ne_src, shapes_src = shape_collate_fn(items_src)
    V_tgt, L_tgt, nv_tgt, ne_tgt, shapes_tgt = shape_collate_fn(items_tgt)

    V_src = V_src.to(device)
    L_src = L_src.to(device)
    nv_src = nv_src.to(device)
    ne_src = ne_src.to(device)
    V_tgt = V_tgt.to(device)
    L_tgt = L_tgt.to(device)
    nv_tgt = nv_tgt.to(device)
    ne_tgt = ne_tgt.to(device)

    resample_M = cfg.get("resample_M", None)
    if resample_M is not None:
        M = int(resample_M)
        V_src, L_src, nv_src, ne_src = resample_uniform_polyline(V_src, L_src, nv_src, M)
        V_tgt, L_tgt, nv_tgt, ne_tgt = resample_uniform_polyline(V_tgt, L_tgt, nv_tgt, M)

    return (
        V_src,
        L_src,
        nv_src,
        ne_src,
        V_tgt,
        L_tgt,
        nv_tgt,
        ne_tgt,
        shapes_src,
        shapes_tgt,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="export_harness_comparison")
def main(cfg: DictConfig) -> None:
    device = _resolve_device(cfg.device)
    num_samples = int(cfg.num_samples)
    panel_size_in = float(cfg.panel_size_in)
    linewidth = float(cfg.linewidth)
    fmt = str(cfg.fmt)

    if bool(cfg.get("stable_output", False)):
        repo_root = Path(__file__).resolve().parents[1]
        out_root = str(repo_root / cfg.output_dir)
    elif os.path.isabs(cfg.output_dir):
        out_root = cfg.output_dir
    else:
        out_root = os.path.join(HydraConfig.get().runtime.output_dir, cfg.output_dir)

    (
        V_src,
        L_src,
        nv_src,
        ne_src,
        V_tgt,
        L_tgt,
        nv_tgt,
        ne_tgt,
        shapes_src,
        shapes_tgt,
    ) = _load_eval_batch(cfg, num_samples, device)

    initial_dir = os.path.join(out_root, "init")
    raw_initial = save_src_tgt_pair_cells(
        V_src.cpu(),
        L_src.cpu(),
        nv_src.cpu(),
        ne_src.cpu(),
        V_tgt.cpu(),
        L_tgt.cpu(),
        nv_tgt.cpu(),
        ne_tgt.cpu(),
        initial_dir,
        num_pairs=num_samples,
        panel_size_in=panel_size_in,
        fmt=fmt,
        linewidth=linewidth,
    )
    initial_paths = []
    for i, src_path in enumerate(raw_initial):
        dst_path = os.path.join(initial_dir, f"{i:03d}.{fmt}")
        os.replace(src_path, dst_path)
        initial_paths.append(dst_path)
    print(f"wrote {len(initial_paths)} initial panels to {initial_dir}")

    for run_cfg in cfg.runs:
        label = str(run_cfg.label)
        run_dir = os.path.join(out_root, label)
        os.makedirs(run_dir, exist_ok=True)

        matcher = instantiate(run_cfg.matcher)
        scenario = instantiate(run_cfg.scenario)

        V_final = scenario.run(
            (V_src, L_src, nv_src, ne_src),
            (V_tgt, L_tgt, nv_tgt, ne_tgt),
            matcher,
        )
        if isinstance(V_final, tuple):
            V_final = V_final[0]

        raw_final = save_src_tgt_pair_cells(
            V_final.cpu(),
            L_src.cpu(),
            nv_src.cpu(),
            ne_src.cpu(),
            V_tgt.cpu(),
            L_tgt.cpu(),
            nv_tgt.cpu(),
            ne_tgt.cpu(),
            run_dir,
            num_pairs=num_samples,
            panel_size_in=panel_size_in,
            fmt=fmt,
            linewidth=linewidth,
        )
        final_paths = []
        for i, src_path in enumerate(raw_final):
            dst_path = os.path.join(run_dir, f"{i:03d}.{fmt}")
            os.replace(src_path, dst_path)
            final_paths.append(dst_path)

        print(f"[{label}] wrote {len(final_paths)} final panels to {run_dir}")
        for i in range(min(3, num_samples)):
            print(
                f"  sample {i}: {shapes_src[i]} → {shapes_tgt[i]}  "
                f"initial={initial_paths[i]}  final={final_paths[i]}"
            )
        if num_samples > 3:
            print(f"  ... ({num_samples - 3} more samples)")


if __name__ == "__main__":
    main()
