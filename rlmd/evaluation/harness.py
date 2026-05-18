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
    V, L, lengths, shapes = batch
    return V.to(device), L.to(device), lengths.to(device), shapes


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


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
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        for batch_src, batch_tgt in tqdm(zip(loader_src, loader_tgt),
                                         total=len(loader_src),
                                         desc="harness"):
            V_src, L_src, nv_src, shapes_src = _to_device(batch_src, device)
            V_tgt, L_tgt, nv_tgt, shapes_tgt = _to_device(batch_tgt, device)

            V_final = scenario.run(
                (V_src, L_src, nv_src),
                (V_tgt, L_tgt, nv_tgt),
                matcher,
            )

            with torch.no_grad():
                poly_pred = (V_final, L_src, nv_src)
                poly_tgt = (V_tgt, L_tgt, nv_tgt)
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
