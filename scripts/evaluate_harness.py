"""Evaluation harness CLI (Hydra). Set ``visualize_deformations=true`` in config or on the command line to write per-batch PNGs (same layout as ``scripts/deform_polylines.py``)."""

import hydra
from omegaconf import DictConfig

from rlmd.evaluation.harness import run


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_harness")
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
