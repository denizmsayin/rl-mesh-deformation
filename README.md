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

## TODOs (17-04)

Run the pytorch3d notebook: https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb

View results with some software like MeshLab, try to understand what's going on, fiddle with it a little!

### Data Generation

Our data representation is two arrays for each shape: vertices `(V, 2)` and line segments `(L, 2)`. 

Initially, we want to be able to generate lots of simple 2D shapes quickly. They each might have their own configuration parameters. For a circle, this might be the number of points we use to generate its shape. The core generation should be kept as simple as possible, most basic variations can be applied via rotation, scaling and translation tranformations. e.g. a basic circle should always be a unit circle. 

We might want some function similar to this:

```py
generate_shapes(num_shapes, shape_config, scale_range, rotate_range, translate_range)
```

`shape_config` could be a dict depending on the object. `{"name": "circle", "num_points": 50}` and `{"name": "hexagon"}` could work. 

The ranges will be tuple ranges for randomization of data. e.g. a scale range of `(1.0, 1.0)` would mean we do not change the shape, but `(0.5, 2.0)` would mean we pick a random value between 0.5 and 2.0 for each generated shape and scale it. 

Rotate could be in degrees for convenience. 

Feel free to complicate or simplify as necessary!

### Visualization

We need to be able to easily visualize our intermediate results for debugging and evaluation. Something abstract like:

```py
visualize_data(name, source, target, matches)
```

`source` and `target` will be tuples containing `(V, L)` arrays defining the shape. `matches` will be an array of shape `(V_s,)`, containing the index in `V_t` each source vertex matched to. `name` can be an identifier for the files the visualizations will be saved in, as more than one file might be necessary. 

At its simplest, we could generate some PNG images that draw both shapes and arrows between the matches. Or color the source with a color gradient and put the same colors in the target. Some heuristics to automatically deal with large numbers points like only drawing some representative arrows etc. would be nice. 

It would be nice to also have something like `visualize_sequence(names)` that will use the data saved for several visualizations and generate some combined visualization, like a GIF of how the process changed. 

