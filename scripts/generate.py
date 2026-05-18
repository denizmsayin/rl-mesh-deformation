import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from rlmd.data.generation import ShapeGenerator


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig):
    print("Generating shapes with the following configuration:")
    print(cfg)

    shapes = {
        "circle": {"shape": "circle", "num_points": 60, "percentage": 0.30},
        "hexagon": {"shape": "hexagon", "num_points": 60, "percentage": 0.25},
        "triangle": {"shape": "triangle", "num_points": 60, "percentage": 0.20},
        "star_5": {"shape": "star", "num_points": 60, "n_tips": 5, "inner_radius": 0.45, "percentage": 0.15},
        "star_12": {"shape": "star", "num_points": 2500, "n_tips": 12, "inner_radius": 0.6, "percentage": 0.10},
    }

    shape_generator = ShapeGenerator()

    device = ShapeGenerator._cfg_get(cfg, "device", "auto")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = int(ShapeGenerator._cfg_get(cfg, "seed", 123))
    preview_samples_per_shape = int(ShapeGenerator._cfg_get(cfg, "preview_samples_per_shape", 20))
    preview_output_file = ShapeGenerator._cfg_get(cfg, "preview_output_file", "data_generation/generated_shapes.png")

    dataset_N = int(ShapeGenerator._cfg_get(cfg, "N", 100_000))
    batch_size = int(ShapeGenerator._cfg_get(cfg, "batch_size", 4096))
    out_dir = ShapeGenerator._cfg_get(cfg, "out_dir", "data_generation/generated_dataset")
    save_mode = ShapeGenerator._cfg_get(cfg, "save_mode", "params")
    overwrite = bool(ShapeGenerator._cfg_get(cfg, "overwrite", True))

    preview_batch = shape_generator.generate_mixture_batch_torch(
        shapes=shapes,
        transform_cfg=cfg,
        samples_per_shape=preview_samples_per_shape,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )

    preview_batch.plot(
        max_per_shape=preview_samples_per_shape,
        title=f"{preview_samples_per_shape} augmentations per shape",
        output_file=preview_output_file,
        show=False,
    )

    plt.close()

    print(f"Saved {preview_output_file}")

    manifest = shape_generator.generate_to_disk_torch(
        shapes=shapes,
        transform_cfg=cfg,
        out_dir=out_dir,
        N=dataset_N,
        batch_size=batch_size,
        seed=seed,
        device=device,
        dtype=torch.float32,
        save_mode=save_mode,
        overwrite=overwrite,
    )

    print("Dataset generated successfully!")
    print(f"Number of shards: {len(manifest['shards'])}")


if __name__ == "__main__":
    main()
