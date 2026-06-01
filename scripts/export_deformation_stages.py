"""Export evenly spaced deformation snapshots as publication PDFs for one eval sample.

Re-runs the configured scenario with frame recording so exports stay reproducible
without reading an MP4. Styling matches ``scripts/export_dataset_pairs.py`` with
a thinner default linewidth.

Example (matches a typical evaluate_harness run):

  pixi run python scripts/export_deformation_stages.py \\
    sample_idx=5 resample_M=64 record_every=5 \\
    scenario.num_iters=300 scenario.w_chamfer=1.0
"""

import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rlmd.dataset import shape_collate_fn
from rlmd.evaluation.harness import _build_dataset, _resolve_device
from rlmd.ops import resample_uniform_polyline
from rlmd.visualization.visualize import evenly_spaced_indices, save_deformation_stage_pdfs


def _resolve_frame_indices(cfg: DictConfig, num_recorded: int) -> list[int]:
    if cfg.get("frame_indices") is not None:
        return [int(i) for i in cfg.frame_indices]
    n = int(cfg.num_frames)
    if n < 1:
        raise ValueError("num_frames must be at least 1.")
    return evenly_spaced_indices(num_recorded, n).tolist()


@hydra.main(version_base=None, config_path="../configs", config_name="export_deformation_stages")
def main(cfg: DictConfig) -> None:
    device = _resolve_device(cfg.device)
    sample_idx = int(cfg.sample_idx)

    dataset_src = _build_dataset(cfg.dataset_src, cfg.eval_num_samples, cfg.seed_src)
    dataset_tgt = _build_dataset(cfg.dataset_tgt, cfg.eval_num_samples, cfg.seed_tgt)
    if sample_idx >= len(dataset_src) or sample_idx >= len(dataset_tgt):
        raise ValueError(
            f"sample_idx={sample_idx} out of range for eval subset size "
            f"{len(dataset_src)}."
        )

    batch_src = shape_collate_fn([dataset_src[sample_idx]])
    batch_tgt = shape_collate_fn([dataset_tgt[sample_idx]])
    V_src, L_src, nv_src, ne_src, shapes_src = batch_src
    V_tgt, L_tgt, nv_tgt, ne_tgt, shapes_tgt = batch_tgt

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

    matcher = instantiate(cfg.matcher)
    scenario = instantiate(cfg.scenario)

    record_every = int(cfg.record_every)
    V_final, frames, _ = scenario.run(
        (V_src, L_src, nv_src, ne_src),
        (V_tgt, L_tgt, nv_tgt, ne_tgt),
        matcher,
        record_every=record_every,
        record_max_batch=1,
    )
    del V_final

    frames_k = frames[:, 0].cpu()
    frame_indices = _resolve_frame_indices(cfg, frames_k.shape[0])

    out_dir = cfg.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(HydraConfig.get().runtime.output_dir, out_dir)

    paths = save_deformation_stage_pdfs(
        frames_k,
        L_src.cpu(),
        nv_src.cpu(),
        ne_src.cpu(),
        V_tgt.cpu(),
        L_tgt.cpu(),
        nv_tgt.cpu(),
        ne_tgt.cpu(),
        out_dir,
        frame_indices=frame_indices,
        sample_idx=sample_idx,
        record_every=record_every,
        num_iters=int(cfg.scenario.num_iters),
        panel_size_in=float(cfg.panel_size_in),
        fmt=str(cfg.fmt),
        linewidth=float(cfg.linewidth),
    )

    src_shape = shapes_src[0]
    tgt_shape = shapes_tgt[0]
    print(
        f"sample {sample_idx}: {src_shape} → {tgt_shape} "
        f"({cfg.dataset_src.name} → {cfg.dataset_tgt.name})"
    )
    print(f"recorded {frames_k.shape[0]} frames, exported indices {frame_indices}")
    print(f"wrote {len(paths)} files to {out_dir}")
    for p in paths:
        print(f"  {p}")
    print(f"scenario: {OmegaConf.to_yaml(cfg.scenario, resolve=True).strip()}")


if __name__ == "__main__":
    main()
