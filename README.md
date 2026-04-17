# rl-mesh-deformation

## Environment Installation

Install `pixi`: https://pixi.prefix.dev/latest/installation/

```sh
pixi install
```

This will resolve and install the environment in the local project directory. You can then activate the environment using `pixi shell`. Or, you can run commands with the environment without activating it by using `pixi run <command>`. 

Environment specifications are stored inside `pixi.toml`. There's also a lockfile `pixi.lock` that `pixi` itself generates after environment resolution for pinning package versions. **Both files must be committed and up-to-date in the repository to ensure environment consistency.**
