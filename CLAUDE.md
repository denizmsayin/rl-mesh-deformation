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

### Data flow conventions

- Long-format CSV is the canonical output (per-sample, per-metric rows — `notes.md`).
- The harness expects `dataset_src` and `dataset_tgt` to yield same-shaped batches and pairs them positionally; `eval_num_samples` truncates via a seeded random `Subset`.
- Datasets live under `data_generation/generated_dataset/<dataset_name>/` by default; override the root with the `RLMD_DATASET` env var.
- Outputs (CSVs, PNGs, Hydra run dirs) land under `outputs/` — `hydra.run.dir` is configured per script.
