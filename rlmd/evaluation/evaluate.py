import os
import json
import bisect
from collections import OrderedDict
from omegaconf import DictConfig
import hydra 
from hydra.utils import instantiate 
import csv

import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
from hydra.core.hydra_config import HydraConfig

from  evaluation_utils import chamfer, segments_std
from rlmd.dataset import ShapeDiskDataset, shape_collate_fn


def make_two_different_shape_datasets(
    dataset_folder,
    n_per_dataset=None,
    shape_names=None,
    seed=0,
    cache_size=8,
    dtype=torch.float32,
):
    base_dataset = ShapeDiskDataset(
        dataset_folder=dataset_folder,
        max_samples=None,
        shape_names=shape_names,
        cache_size=cache_size,
        dtype=dtype,
    )

    total = len(base_dataset)

    if n_per_dataset is None:
        n_per_dataset = total // 2

    if 2 * n_per_dataset > total:
        raise ValueError(
            f"Cannot create two datasets of size {n_per_dataset}: "
            f"base dataset contains only {total} elements."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(total, generator=generator)

    indices_1 = indices[:n_per_dataset].tolist()
    indices_2 = indices[n_per_dataset:2 * n_per_dataset].tolist()

    dataset_1 = Subset(base_dataset, indices_1)
    dataset_2 = Subset(base_dataset, indices_2)

    return dataset_1, dataset_2


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig):
    print("Evaluating with the following configuration:")
    print(cfg)

    device = torch.device(cfg.device if hasattr(cfg, "device") else "cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model)
    model = model.to(device)
    model.eval()

    dataset, dataset_target = make_two_different_shape_datasets(
        dataset_folder=cfg.dataset_folder,
        n_per_dataset=cfg.num_shapes_per_dataset,
        shape_names=cfg.shape_names,
        seed=cfg.seed,
        cache_size=cfg.cache_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=shape_collate_fn,
    )

    dataloader_target = DataLoader(
        dataset_target,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=shape_collate_fn,
    )



    hydra_output_dir = HydraConfig.get().runtime.output_dir

    output_csv_name = cfg.output_csv if hasattr(cfg, "output_csv") else "evaluation_results.csv"
    output_csv = os.path.join(hydra_output_dir, output_csv_name)

    model_name = cfg.model._target_ if hasattr(cfg.model, "_target_") else str(cfg.model)

    file_exists = os.path.exists(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "dataset_name",
                "model",
                "shape_in",
                "shape_out",
                "chf",
                "std_dev",
            ])

        with torch.no_grad():
            for batch_idx, (batch_in, batch_target) in enumerate(zip(dataloader, dataloader_target)):

                vertices_in, edges_in, lengths_in, shapes_in = batch_in
                vertices_target, edges_target, lengths_target, shapes_target = batch_target

                vertices_in = vertices_in.to(device)
                edges_in = edges_in.to(device)
                lengths_in = lengths_in.to(device)

                vertices_target = vertices_target.to(device)
                edges_target = edges_target.to(device)
                lengths_target = lengths_target.to(device)

                pred_vertices, pred_edges = model(vertices_in, edges_in) # I don't know how the model will be 

                if pred_edges.ndim == 2:
                    pred_edges = pred_edges.unsqueeze(0).expand(pred_vertices.shape[0], -1, -1)

                pred_edges = pred_edges.to(device)

                B = pred_vertices.shape[0]

                for i in range(B):
                    pred_vertices_i = pred_vertices[i:i + 1]
                    vertices_target_i = vertices_target[i:i + 1]

                    pred_edges_i = pred_edges[i:i + 1]

                    lengths_in_i = lengths_in[i:i + 1]
                    lengths_target_i = lengths_target[i:i + 1]

                    chf_i, _ = chamfer(
                        x=pred_vertices_i,
                        y=vertices_target_i,
                        x_lengths=lengths_in_i,
                        y_lengths=lengths_target_i,
                        weights=None,
                        batch_reduction="mean",
                        point_reduction="mean",
                    )

                    std_dev_i = segments_std(
                        points=pred_vertices_i,
                        connect=pred_edges_i,
                        lengths=lengths_in_i,
                    )

                    writer.writerow([
                        cfg.dataset_name,
                        model_name,
                        shapes_in[i],
                        shapes_target[i],
                        float(chf_i.detach().cpu()),
                        float(std_dev_i.detach().cpu()),
                    ])

    
    

if __name__ == "__main__":
    main()





