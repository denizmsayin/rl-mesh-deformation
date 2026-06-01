"""Export one fixed-size PDF per (src, tgt) pair for LaTeX subfigures."""

import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from rlmd.evaluation.harness import _build_dataset, _make_loader
from rlmd.visualization.visualize import save_src_tgt_pair_cells


@hydra.main(version_base=None, config_path="../configs", config_name="export_dataset_pairs")
def main(cfg: DictConfig) -> None:
    dataset_src = _build_dataset(cfg.dataset_src, cfg.eval_num_samples, cfg.seed)
    dataset_tgt = _build_dataset(cfg.dataset_tgt, cfg.eval_num_samples, cfg.seed)
    if len(dataset_src) != len(dataset_tgt):
        raise ValueError(
            f"src and tgt datasets must have equal size (got {len(dataset_src)} vs "
            f"{len(dataset_tgt)})."
        )

    n = int(cfg.num_pairs)
    if n > len(dataset_src):
        raise ValueError(f"num_pairs={n} exceeds dataset size {len(dataset_src)}.")

    loader_src = _make_loader(dataset_src, cfg.batch_size, cfg.num_workers, cfg.pin_memory)
    loader_tgt = _make_loader(dataset_tgt, cfg.batch_size, cfg.num_workers, cfg.pin_memory)

    batch_src = next(iter(loader_src))
    batch_tgt = next(iter(loader_tgt))
    V_src, L_src, nv_src, ne_src, _ = batch_src
    V_tgt, L_tgt, nv_tgt, ne_tgt, _ = batch_tgt

    out_dir = cfg.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(HydraConfig.get().runtime.output_dir, out_dir)

    paths = save_src_tgt_pair_cells(
        V_src[:n].cpu(),
        L_src[:n].cpu(),
        nv_src[:n].cpu(),
        ne_src[:n].cpu(),
        V_tgt[:n].cpu(),
        L_tgt[:n].cpu(),
        nv_tgt[:n].cpu(),
        ne_tgt[:n].cpu(),
        out_dir,
        num_pairs=n,
        panel_size_in=float(cfg.panel_size_in),
        fmt=str(cfg.fmt),
        linewidth=float(cfg.linewidth),
    )
    print(f"wrote {len(paths)} files to {out_dir}  ({cfg.dataset_src.name} → {cfg.dataset_tgt.name})")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
