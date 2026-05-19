"""REINFORCE training for a learned polyline matcher.

Per batch:
  1. Resample (V_src, V_tgt) to M-vertex uniform polylines.
  2. Sample matches with a stochastic learned matcher; keep log_prob.
  3. Run SgdFixedMatchScenario with the frozen action (FixedMatcher adapter).
  4. Reward = -symmetric Chamfer(V_final, V_tgt).
  5. Loss = -(advantage.detach() * log_prob).mean() - entropy_coef * entropy.mean().
  6. Optimizer step on the matcher's feature-extractor params.

Hydra entry; config at configs/train_matcher.yaml.
"""
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rlmd.dataset import shape_collate_fn
from rlmd.evaluation.matchers import FixedMatcher
from rlmd.evaluation.metrics import ChamferMetric
from rlmd.ops import resample_uniform_polyline


def _resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _to_device(batch, device):
    V, L, lengths, shapes = batch
    return V.to(device), L.to(device), lengths.to(device), shapes


def _make_loader(dataset, batch_size, num_workers, pin_memory, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=shape_collate_fn,
        generator=g,
    )


class _Baseline:
    """Scalar baseline applied to (B,) rewards. Supported types: 'none', 'ema'."""

    def __init__(self, cfg: DictConfig):
        self.type = str(cfg.get("type", "none"))
        if self.type == "ema":
            self.momentum = float(cfg.get("momentum", 0.99))
            self.value = None
        elif self.type != "none":
            raise ValueError(f"unknown baseline type: {self.type!r}")

    def __call__(self, R: torch.Tensor) -> torch.Tensor:
        if self.type == "none":
            return torch.zeros_like(R)
        # ema
        batch_mean = R.detach().mean().item()
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * batch_mean
        return torch.full_like(R, self.value)


def train(cfg: DictConfig) -> str:
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    ds_src = instantiate(cfg.dataset_src.dataset)
    ds_tgt = instantiate(cfg.dataset_tgt.dataset)
    n = min(len(ds_src), len(ds_tgt))
    if cfg.get("train_num_samples") is not None:
        n = min(n, int(cfg.train_num_samples))
    if n < len(ds_src):
        ds_src = Subset(ds_src, range(n))
    if n < len(ds_tgt):
        ds_tgt = Subset(ds_tgt, range(n))

    loader_src = _make_loader(ds_src, cfg.batch_size, cfg.num_workers, cfg.pin_memory,
                              shuffle=True, seed=cfg.seed)
    loader_tgt = _make_loader(ds_tgt, cfg.batch_size, cfg.num_workers, cfg.pin_memory,
                              shuffle=True, seed=cfg.seed + 1)

    matcher = instantiate(cfg.matcher)
    matcher.feature_extractor.to(device)
    matcher.feature_extractor.train()

    optimizer = instantiate(cfg.optimizer, params=matcher.feature_extractor.parameters())

    scenario = instantiate(cfg.scenario)

    chamfer = ChamferMetric(num_samples=int(cfg.reward_num_samples),
                            point_reduction="mean", norm=2, with_normals=False)

    baseline = _Baseline(cfg.baseline)

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    log_path = os.path.join(output_dir, "train_log.csv")
    ckpt_path = os.path.join(output_dir, "matcher.pt")
    OmegaConf.save(cfg, os.path.join(output_dir, "resolved_config.yaml"), resolve=True)

    step = 0
    with open(log_path, "w") as logf:
        logf.write("step,epoch,batch,reward_mean,reward_std,loss,entropy,baseline\n")

        for epoch in range(int(cfg.num_epochs)):
            n_batches = min(len(loader_src), len(loader_tgt))
            iterator = tqdm(enumerate(zip(loader_src, loader_tgt)),
                            total=n_batches,
                            desc=f"epoch {epoch+1}/{cfg.num_epochs}")
            for batch_i, (batch_src, batch_tgt) in iterator:
                V_src, L_src, nv_src, _ = _to_device(batch_src, device)
                V_tgt, L_tgt, nv_tgt, _ = _to_device(batch_tgt, device)

                V_src_r, L_src_r, nv_src_r = resample_uniform_polyline(
                    V_src, L_src, nv_src, int(cfg.M))
                V_tgt_r, L_tgt_r, nv_tgt_r = resample_uniform_polyline(
                    V_tgt, L_tgt, nv_tgt, int(cfg.M))

                matchings, log_prob, entropy = matcher(V_src_r, nv_src_r, V_tgt_r, nv_tgt_r)

                V_final = scenario.run(
                    (V_src_r, L_src_r, nv_src_r),
                    (V_tgt_r, L_tgt_r, nv_tgt_r),
                    FixedMatcher(matchings),
                )

                with torch.no_grad():
                    out = chamfer((V_final, L_src_r, nv_src_r),
                                  (V_tgt_r, L_tgt_r, nv_tgt_r))
                    R = -out["chamfer_sym"]                       # (B,)
                    b = baseline(R)                                # (B,)
                    advantage = R - b

                loss = -(advantage.detach() * log_prob).mean() \
                       - float(cfg.entropy_coef) * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                logf.write(
                    f"{step},{epoch},{batch_i},"
                    f"{R.mean().item():.6g},{R.std().item():.6g},"
                    f"{loss.item():.6g},{entropy.mean().item():.6g},"
                    f"{b[0].item():.6g}\n"
                )
                logf.flush()

                iterator.set_postfix({
                    "R": f"{R.mean().item():.4f}",
                    "ent": f"{entropy.mean().item():.3f}",
                    "loss": f"{loss.item():.4f}",
                })

            torch.save({
                "feature_extractor_state_dict": matcher.feature_extractor.state_dict(),
                "temperature": matcher.temperature,
                "M": int(cfg.M),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)

    return ckpt_path


@hydra.main(version_base=None, config_path="../configs", config_name="train_matcher")
def main(cfg: DictConfig) -> None:
    out = train(cfg)
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()
