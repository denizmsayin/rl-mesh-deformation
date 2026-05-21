# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Dependencies are managed by `pixi` (see `pixi.toml`). Two environments are defined:

- Default (CPU): `pixi shell` / `pixi run <cmd>` — uses `pytorch==2.10.0` from conda-forge.
- CUDA (linux-64 only, CUDA 12.8): `pixi shell -e cuda` / `pixi run -e cuda <cmd>` — uses official PyTorch CUDA wheels.

`pytorch3d` is built from source via `pixi run pytorch3d-install` (or `pixi run -e cuda pytorch3d-install`). Both `pixi.toml` and `pixi.lock` must stay committed and in sync.

The package itself (`rlmd`) is installed editable into the env.

## Common commands

```sh
# Tests (pytest config in pyproject.toml — testpaths=["tests"], pythonpath=["."])
pixi run pytest                       # full suite
pixi run pytest tests/test_segment_std.py            # single file
pixi run pytest tests/test_segment_std.py::test_name # single test

# Data generation (Hydra entrypoint, config in configs/generate.yaml)
pixi run python scripts/generate.py                          # uses defaults
pixi run python scripts/generate.py dataset=shapes_basic dataset_name=mix N=50000

# Evaluation harness (Hydra, config in configs/evaluate_harness.yaml)
pixi run python scripts/evaluate_harness.py
pixi run python scripts/evaluate_harness.py visualize_deformations=true scenario=sgd_quick

# Standalone deformation demo (no Hydra)
pixi run python scripts/deform_polylines.py

# Generated dataset location can be overridden by env var
RLMD_DATASET=/path/to/datasets pixi run python scripts/evaluate_harness.py
```

`scripts/bubble.sh` (and the top-level `bubble.sh`) wrap Claude in a `bwrap` sandbox — not part of the project pipeline.

## Architecture

The project explores 2D polyline-shape deformation: a *source* polyline is deformed toward a *target* polyline via gradient-based optimization, with the eventual goal of replacing the optimizer with a learned policy. Shapes are represented as a pair of tensors throughout: `V (B, N, 2)` vertex coords and `L (B, M, 2)` edge index pairs, plus a `num_verts (B,)` length tensor since batches are zero-padded.

### Package layout (`rlmd/`)

- `data/generation.py` — `ShapeGenerator` produces parametric 2D shapes (circle, ellipse, star, hexagon, …) with random scale/rotate/translate augmentations. Used both online (preview/demo) and offline (sharded on-disk datasets).
- `dataset.py` — `ShapeDiskDataset` reads the on-disk manifest+shards layout produced by `ShapeGenerator.generate_to_disk_torch`. `shape_collate_fn` does the variable-length padding to `(V, L, lengths, shapes)`.
- `batching.py` — `pad_polylines` for ad-hoc padding outside the Dataset path.
- `ops/` — differentiable polyline primitives:
  - `sampling.sample_points_from_polylines` (length-weighted edge sampling),
  - `distance.knn_match` + `distance_loss` (Chamfer-style matching loss, bidirectional supported),
  - `losses.polyline_edge_loss`, `polyline_laplacian_smoothing`, `polyline_normal_consistency`.
- `models/polygon_cnn.py` — `PolygonCNN`, the candidate learned model.
- `evaluation/` — Hydra-driven evaluation harness:
  - `harness.run(cfg)` is the entry; iterates `(dataset_src, dataset_tgt)` pairs through a `matcher`, runs a `scenario` to produce `V_final`, then computes a list of `metrics` and writes a long-format CSV.
  - `matchers/` (`knn_3d`), `scenarios/` (`sgd` — gradient descent baseline), `metrics/` (`chamfer`, `segment_std`, `self_intersection`).
- `visualization/visualize.py` — `plot_polylines_initial_vs_final` and friends (referenced from harness when `visualize_deformations=true`).

### Configs (`configs/`)

Hydra is the only config system. The top-level files (`generate.yaml`, `evaluate.yaml`, `evaluate_harness.yaml`, `classify.yaml`, `decenter.yaml`, `generate_r.yaml`) each compose from group dirs:

- `dataset/`, `dataset/transform/` — feed `scripts/generate.py`
- `dataset_src/`, `dataset_tgt/`, `matcher/`, `metric/`, `scenario/`, `model/`, `train/` — feed the harness and training scripts

Components are instantiated via `hydra.utils.instantiate` with `_target_` pointing at concrete classes in `rlmd/`. To add a new metric/matcher/scenario, write the class, then drop a yaml under the matching group dir.

### Adding a new shape dataset (training stream + eval disk set)

A "dataset" for the matcher training / harness pipeline is a `(dataset_src, dataset_tgt)` pair. Each side has two flavors that should be kept in sync:

- **Stream** (`configs/dataset_src/*_stream.yaml`, `configs/dataset_tgt/*_stream.yaml`) — `_target_: rlmd.data.online.OnlineShapeSampler`. Used by training (`scripts/train_matcher.py`). Each `next_batch` call picks **one** spec from `shape_specs` and returns a homogeneous batch, so a multi-spec stream only mixes shapes across batches.
- **Set** (`configs/dataset_src/*_set.yaml`, `configs/dataset_tgt/*_set.yaml`) — `_target_: rlmd.dataset.ShapeDiskDataset`. Reads pre-generated shards from `${RLMD_DATASET}/<name>/` (default root `data_generation/generated_dataset/`). Used by the eval harness and by the periodic eval inside `train_matcher.py` (the `eval.dataset_src` / `eval.dataset_tgt` defaults).

To add a new pair (worked example: `triangle_centered` to match the existing `circle_centered`):

1. **Generator group entry** — `configs/dataset/<name>_only.yaml`, declaring shape kwargs. Pair it with one of the existing `configs/dataset/transform/*.yaml` files (e.g. `centered`, `translated`, `scaled`). Example `configs/dataset/triangle_only.yaml`:
   ```yaml
   shapes:
     triangle:
       shape: triangle
       num_points: 60
       percentage: 1.0
   ```
2. **Generate the disk shards** — writes `${RLMD_DATASET}/<dataset_name>/manifest.json` + per-shape shards:
   ```sh
   pixi run python scripts/generate.py \
     dataset=triangle_only dataset/transform=centered \
     dataset_name=triangle_centered_set N=2000
   ```
3. **Disk reader config** — `configs/dataset_src/<name>_set.yaml` or `configs/dataset_tgt/<name>_set.yaml`, pointing `dataset_folder` at the generated subdir and listing the spec instance names in `shape_names` (the keys from step 1):
   ```yaml
   name: triangle_centered_set
   dataset:
     _target_: rlmd.dataset.ShapeDiskDataset
     dataset_folder: ${oc.env:RLMD_DATASET,data_generation/generated_dataset}/triangle_centered_set
     shape_names: ["triangle"]
     cache_size: 8
   ```
4. **Stream config** — `configs/dataset_src/<name>_stream.yaml` or `configs/dataset_tgt/<name>_stream.yaml`. The `transform:` block must match what step 2 used so stream and set actually sample from the same distribution; the `shape_specs` list mirrors the `shapes:` map from step 1:
   ```yaml
   name: triangle_centered_stream
   source:
     _target_: rlmd.data.online.OnlineShapeSampler
     shape_specs:
       - {shape: triangle, num_points: 60}
     transform:
       translation_range: [0.0, 0.0]
       scale_range: [2.0, 2.0]
       rotation_range: [0.0, 0.0]
       isotropic_scale: true
     seed: 1
   ```
5. **Use it from a script**, either by editing the `defaults:` block of the consuming top-level config (e.g. `configs/train_matcher.yaml`) or by overriding on the command line. `train_matcher.yaml` brings the eval datasets in under a different package, so eval overrides need the `@`-syntax:
   ```sh
   pixi run -e cuda python scripts/train_matcher.py \
     dataset_src=circle_centered_stream \
     dataset_tgt=triangle_centered_stream \
     dataset_src@eval.dataset_src=circle_centered_set \
     dataset_tgt@eval.dataset_tgt=triangle_centered_set
   ```

Notes:
- Keep stream and set transforms identical for any pair that's meant to back the same experiment — silent drift here makes train-vs-eval reward comparisons meaningless.
- `OnlineShapeSampler` batches are homogeneous per call. If you list multiple shapes in a stream config, a single batch built at startup (e.g. the periodic-eval streamed batch in `train_matcher.py`) will only cover one of them; build multiple batches to span the mixture.
- `RLMD_DATASET` overrides the dataset root for both generation and reading.

### Data flow conventions

- Long-format CSV is the canonical output (per-sample, per-metric rows — `notes.md`).
- The harness expects `dataset_src` and `dataset_tgt` to yield same-shaped batches and pairs them positionally; `eval_num_samples` truncates via a seeded random `Subset`.
- Datasets live under `data_generation/generated_dataset/<dataset_name>/` by default; override the root with the `RLMD_DATASET` env var.
- Outputs (CSVs, PNGs, Hydra run dirs) land under `outputs/` — `hydra.run.dir` is configured per script.
