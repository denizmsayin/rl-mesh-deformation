import csv
import os
from typing import List, Optional

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rlmd.dataset import shape_collate_fn
from rlmd.ops import resample_uniform_polyline
from rlmd.visualization.visualize import (
    plot_polylines_initial_vs_final,
    render_deformation_video,
    save_deformation_cells,
)


CSV_HEADER = [
    "run_id",
    "sample_idx",
    "src_shape",
    "tgt_shape",
    "dataset_src",
    "dataset_tgt",
    "matcher",
    "scenario",
    "metric",
    "value",
]


def _build_dataset(cfg_entry: DictConfig, eval_num_samples: Optional[int],
                   seed: int):
    dataset = instantiate(cfg_entry.dataset)
    total = len(dataset)
    if eval_num_samples is None or eval_num_samples >= total:
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=g)[:eval_num_samples].tolist()
    return Subset(dataset, indices)


def _make_loader(dataset, batch_size: int, num_workers: int,
                 pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=shape_collate_fn,
    )


def _to_device(batch, device):
    V, L, num_verts, num_edges, shapes = batch
    return (V.to(device), L.to(device), num_verts.to(device),
            num_edges.to(device), shapes)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _deformation_vis_tensors(
    V_src: torch.Tensor,
    V_final: torch.Tensor,
    L_src: torch.Tensor,
    nv_src: torch.Tensor,
    ne_src: torch.Tensor,
    V_tgt: torch.Tensor,
    L_tgt: torch.Tensor,
    nv_tgt: torch.Tensor,
    ne_tgt: torch.Tensor,
    *,
    max_batch: Optional[int] = None,
):
    """Return plot inputs on CPU; optionally keep only the first ``max_batch`` rows."""
    if max_batch is not None:
        V_src = V_src[:max_batch]
        V_final = V_final[:max_batch]
        L_src = L_src[:max_batch]
        nv_src = nv_src[:max_batch]
        ne_src = ne_src[:max_batch]
        V_tgt = V_tgt[:max_batch]
        L_tgt = L_tgt[:max_batch]
        nv_tgt = nv_tgt[:max_batch]
        ne_tgt = ne_tgt[:max_batch]
    return (
        V_src.detach().cpu(),
        V_final.detach().cpu(),
        L_src.detach().cpu(),
        nv_src.detach().cpu(),
        ne_src.detach().cpu(),
        V_tgt.detach().cpu(),
        L_tgt.detach().cpu(),
        nv_tgt.detach().cpu(),
        ne_tgt.detach().cpu(),
    )


def _write_deformation_batch_figure(
    V_src: torch.Tensor,
    V_final: torch.Tensor,
    L_src: torch.Tensor,
    nv_src: torch.Tensor,
    ne_src: torch.Tensor,
    V_tgt: torch.Tensor,
    L_tgt: torch.Tensor,
    nv_tgt: torch.Tensor,
    ne_tgt: torch.Tensor,
    out_path: str,
    *,
    first_index: int,
    max_batch: Optional[int] = None,
    dpi: int = 150,
) -> None:
    plot_polylines_initial_vs_final(
        *_deformation_vis_tensors(
            V_src,
            V_final,
            L_src,
            nv_src,
            ne_src,
            V_tgt,
            L_tgt,
            nv_tgt,
            ne_tgt,
            max_batch=max_batch,
        ),
        out_path,
        dpi=dpi,
        title_prefix="sample",
        first_index=first_index,
    )


def run(cfg: DictConfig) -> str:
    """Run the evaluation harness. Returns the path to the written CSV."""
    device = _resolve_device(cfg.device)
    print(f"using device: {device}")
    eval_num_samples = cfg.get("eval_num_samples", None)

    dataset_src = _build_dataset(cfg.dataset_src, eval_num_samples, cfg.seed_src)
    dataset_tgt = _build_dataset(cfg.dataset_tgt, eval_num_samples, cfg.seed_tgt)

    if len(dataset_src) != len(dataset_tgt):
        raise ValueError(
            f"src and tgt datasets must have equal size after capping "
            f"(got {len(dataset_src)} vs {len(dataset_tgt)}); check eval_num_samples."
        )

    loader_src = _make_loader(dataset_src, cfg.batch_size, cfg.num_workers, cfg.pin_memory)
    loader_tgt = _make_loader(dataset_tgt, cfg.batch_size, cfg.num_workers, cfg.pin_memory)

    matcher = instantiate(cfg.matcher)
    scenario = instantiate(cfg.scenario)
    metrics: List = [instantiate(m) for m in cfg.metrics]

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    run_id = os.path.basename(output_dir.rstrip("/"))

    output_csv = os.path.join(output_dir, cfg.output_csv)
    file_exists = os.path.exists(output_csv)

    dataset_src_name = cfg.dataset_src.name
    dataset_tgt_name = cfg.dataset_tgt.name
    matcher_name = matcher.name
    scenario_name = scenario.name

    sample_idx = 0
    visualize = bool(cfg.get("visualize_deformations", False))
    vis_dpi = int(cfg.get("vis_dpi", 150))
    vis_save_cells = bool(cfg.get("vis_save_cells", False))
    vis_cells_fmt = str(cfg.get("vis_cells_format", "png"))
    resample_M = cfg.get("resample_M", None)
    if resample_M is not None:
        resample_M = int(resample_M)

    record_cfg = cfg.get("record_deformation", None)
    record_enabled = bool(record_cfg and record_cfg.get("enabled", False))
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        for batch_i, (batch_src, batch_tgt) in tqdm(
                enumerate(zip(loader_src, loader_tgt)),
                total=len(loader_src),
                desc="harness"):
            V_src, L_src, nv_src, ne_src, shapes_src = _to_device(batch_src, device)
            V_tgt, L_tgt, nv_tgt, ne_tgt, shapes_tgt = _to_device(batch_tgt, device)

            if resample_M is not None:
                V_src, L_src, nv_src, ne_src = resample_uniform_polyline(
                    V_src, L_src, nv_src, resample_M)
                V_tgt, L_tgt, nv_tgt, ne_tgt = resample_uniform_polyline(
                    V_tgt, L_tgt, nv_tgt, resample_M)

            record_this_batch = record_enabled and batch_i == 0
            if record_this_batch:
                K_rec = int(record_cfg.first_k)
                V_final, frames, matchings = scenario.run(
                    (V_src, L_src, nv_src, ne_src),
                    (V_tgt, L_tgt, nv_tgt, ne_tgt),
                    matcher,
                    record_every=int(record_cfg.every),
                    record_max_batch=K_rec,
                )
                K_rec = min(K_rec, int(V_src.shape[0]))
                # Pass the source->target matching (first entry) into the
                # video renderer so it can overlay matching arrows. None for
                # scenarios that don't have a single fixed matching.
                match_idx_vis = None
                if matchings is not None and len(matchings) > 0:
                    match_idx_vis = matchings[0].idx_tgt[:K_rec].detach().cpu()
                video_path = os.path.join(output_dir, str(record_cfg.filename))
                render_deformation_video(
                    frames,
                    L_src[:K_rec],
                    nv_src[:K_rec],
                    ne_src[:K_rec],
                    V_tgt[:K_rec],
                    L_tgt[:K_rec],
                    nv_tgt[:K_rec],
                    ne_tgt[:K_rec],
                    video_path,
                    match_idx=match_idx_vis,
                    duration_s=float(record_cfg.duration_s),
                    first_index=sample_idx,
                )
                print(f"wrote deformation video {video_path}")
            else:
                V_final = scenario.run(
                    (V_src, L_src, nv_src, ne_src),
                    (V_tgt, L_tgt, nv_tgt, ne_tgt),
                    matcher,
                )

            if visualize:
                vis_path = os.path.join(output_dir,
                                        f"deformations_batch_{batch_i:05d}.png")
                _write_deformation_batch_figure(
                    V_src,
                    V_final,
                    L_src,
                    nv_src,
                    ne_src,
                    V_tgt,
                    L_tgt,
                    nv_tgt,
                    ne_tgt,
                    vis_path,
                    first_index=sample_idx,
                    dpi=vis_dpi,
                )
                print(f"wrote visualization {vis_path}")

                B_vis = int(V_src.shape[0])
                if B_vis > 8:
                    preview_path = os.path.join(
                        output_dir,
                        f"deformations_batch_{batch_i:05d}_first8.png",
                    )
                    _write_deformation_batch_figure(
                        V_src,
                        V_final,
                        L_src,
                        nv_src,
                        ne_src,
                        V_tgt,
                        L_tgt,
                        nv_tgt,
                        ne_tgt,
                        preview_path,
                        first_index=sample_idx,
                        max_batch=8,
                        dpi=vis_dpi,
                    )
                    print(f"wrote visualization {preview_path}")

                if vis_save_cells:
                    cells_dir = os.path.join(output_dir, "cells")
                    tensors = _deformation_vis_tensors(
                        V_src, V_final, L_src, nv_src, ne_src,
                        V_tgt, L_tgt, nv_tgt, ne_tgt
                    )
                    save_deformation_cells(
                        *tensors,
                        cells_dir,
                        dpi=vis_dpi,
                        fmt=vis_cells_fmt,
                        first_index=sample_idx,
                    )
                    print(f"wrote cells to {cells_dir}")

            with torch.no_grad():
                poly_pred = (V_final, L_src, nv_src, ne_src)
                poly_tgt = (V_tgt, L_tgt, nv_tgt, ne_tgt)
                B = V_final.shape[0]

                metric_values = {}
                for metric in metrics:
                    out = metric(poly_pred, poly_tgt)
                    for sub_name, vec in out.items():
                        metric_values[sub_name] = vec.detach().cpu()

                for i in range(B):
                    for sub_name, vec in metric_values.items():
                        writer.writerow([
                            run_id,
                            sample_idx + i,
                            shapes_src[i],
                            shapes_tgt[i],
                            dataset_src_name,
                            dataset_tgt_name,
                            matcher_name,
                            scenario_name,
                            sub_name,
                            float(vec[i].item()),
                        ])

            sample_idx += B

    OmegaConf.save(cfg, os.path.join(output_dir, "resolved_config.yaml"), resolve=True)
    return output_csv
