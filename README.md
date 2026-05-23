# rl-mesh-deformation

## Environment Installation

Install `pixi`: https://pixi.prefix.dev/latest/installation/

```sh
pixi install
```

This will resolve and install the environment in the local project directory. You can then activate the environment using `pixi shell`. Or, you can run commands with the environment without activating it by using `pixi run <command>`. 

Environment specifications are stored inside `pixi.toml`. There's also a lockfile `pixi.lock` that `pixi` itself generates after environment resolution for pinning package versions. **Both files must be committed and up-to-date in the repository to ensure environment consistency.**

Some libraries are more of a pain to install than others, especially if we have to consider possible GPU usage. Take a look at `[tasks]` under `pixi.toml` for such libraries. We currently have a build-from-source command for pytorch3d, and you can install it by:
```sh
pixi run pytorch3d-install
```

### CUDA environment (linux-64 only)

For running on GPU servers, a separate `cuda` environment is available. It requires CUDA 12.8 and uses the official PyTorch CUDA wheels. Activate it by passing `-e cuda` to any pixi command:

```sh
pixi shell -e cuda
pixi run -e cuda python train.py
pixi run -e cuda pytorch3d-install  # build pytorch3d against the CUDA torch
```

The default CPU environment is unaffected — omitting `-e cuda` always gives you the standard environment.

## Usage

All scripts use Hydra; common knobs can be overridden inline as `key=value`. Outputs land under `outputs/<script>/<date>/<time>/` unless overridden. The dataset root defaults to `data_generation/generated_dataset/` — set `RLMD_DATASET=/path` to point elsewhere.

### Generate a dataset

```sh
# Produce ${RLMD_DATASET}/triangle_centered_set/ with 2000 samples
pixi run python scripts/generate.py \
  dataset=triangle_only dataset/transform=centered \
  dataset_name=triangle_centered_set N=2000
```

Common knobs: `dataset=<group>` (shape spec, see `configs/dataset/`), `dataset/transform=<group>` (augmentation, see `configs/dataset/transform/`), `N` (sample count), `dataset_name` (output subdir), `seed`.

To (re)generate every set referenced by `configs/dataset_{src,tgt}/*_set.yaml` in one go:

```sh
./generate_datasets.sh          # N=2000 per set by default
N=50000 ./generate_datasets.sh  # override sample count
```

### Evaluation harness

Runs `(src, tgt)` pairs through a matcher + scenario and writes a long-format CSV plus optional PNGs.

```sh
# Default: circle_translated_set → any_set with KNN matcher + SGD scenario
pixi run python scripts/evaluate_harness.py

# Swap components, name the run, enable per-sample cell figures
pixi run python scripts/evaluate_harness.py \
  dataset_src=circle_centered_set dataset_tgt=triangle_centered_set \
  scenario=sgd_quick run_name=triangle_quick \
  visualize_deformations=true vis_save_cells=true

# Record an MP4 of the deformation for the first batch
pixi run python scripts/evaluate_harness.py \
  record_deformation.enabled=true record_deformation.first_k=8
```

Common knobs: `dataset_src=` / `dataset_tgt=` (any `*_set` under `configs/dataset_{src,tgt}/`), `matcher=` (`knn_3d`, `learned`, `learned_stochastic`), `scenario=` (`sgd_default`, `sgd_quick`, `sgd_fixed_match`), `eval_num_samples` (subset size), `batch_size`, `device=auto|cpu|cuda`, `resample_M=<int>` (uniform arc-length resample — required when matcher is `learned*`, must match training `M`).

### Train the learned matcher

```sh
# Defaults: src=circle_centered, tgt=triangle_centered, learned_stochastic + sgd_fixed_match
pixi run -e cuda python scripts/train_matcher.py

# Swap datasets (note the @-syntax for eval overrides — see CLAUDE.md for the dataset pair walkthrough)
pixi run -e cuda python scripts/train_matcher.py \
  src=circle_centered tgt=triangle_centered \
  dataset_src@eval.dataset_src=circle_centered_set \
  dataset_tgt@eval.dataset_tgt=triangle_centered_set

# Smaller budget, EMA baseline, periodic eval every 250 steps
pixi run -e cuda python scripts/train_matcher.py \
  total_trajectories=200_000 \
  baseline.type=ema baseline.momentum=0.9 \
  eval.every_steps=250 checkpoint_every_steps=250
```

Common knobs: `total_trajectories` (sample budget), `batch_size`, `M` (resample target — also the eval-time `resample_M`), `optimizer.lr`, `entropy_coef`, `reward.w_chamfer` / `reward.w_normal`, `baseline.type` (`none`, `ema`, `prior`, `chamfer_sgd`), `eval.every_steps` / `eval.compare_to_knn`, `checkpoint_every_steps`.

### Evaluate a trained matcher

Point the harness at the `matcher=learned` config and pass the checkpoint produced by training. `resample_M` must match the `M` used during training (16 by default).

```sh
pixi run -e cuda python scripts/evaluate_harness.py \
  matcher=learned \
  matcher.checkpoint_path=outputs/train_matcher/2026-05-22/12-00-00/matcher.pt \
  dataset_src=circle_centered_set dataset_tgt=triangle_centered_set \
  resample_M=16 run_name=learned_eval
```

Use `matcher=learned_stochastic` to evaluate with the sampling policy instead of argmax. Tweak `matcher.temperature` (and `matcher.override_temperature=true` for `learned`) to sharpen or soften the matching distribution.
