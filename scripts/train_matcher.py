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
from rlmd.evaluation.matchers import FixedMatcher, Knn3dMatcher, LearnedMatcher
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


def _cycle(loader):
    """Infinite iterator over `loader`; reshuffles each pass via the loader's generator."""
    while True:
        for batch in loader:
            yield batch


def _build_eval_subset(dataset, num_samples, seed):
    """Deterministic Subset of `dataset` with `num_samples` items (or fewer)."""
    if num_samples is None or num_samples >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=g)[:int(num_samples)].tolist()
    return Subset(dataset, indices)


def _rollout_reward(matcher, scenario, chamfer, batches_src, batches_tgt, M, device):
    """Run scenario for each pair, return concatenated per-item rewards (-chamfer_sym).

    NOTE: cannot wrap the scenario call in torch.no_grad() because the scenario's
    inner SGD requires gradients on `deform`. Both LearnedMatcher and
    Knn3dMatcher avoid leaking gradients to feature-extractor params (the
    former wraps its forward in no_grad, the latter has no params). Only the
    reward Chamfer is wrapped in no_grad here.
    """
    rewards = []
    for batch_src, batch_tgt in zip(batches_src, batches_tgt):
        V_src, L_src, nv_src, _ = _to_device(batch_src, device)
        V_tgt, L_tgt, nv_tgt, _ = _to_device(batch_tgt, device)
        V_src_r, L_src_r, nv_src_r = resample_uniform_polyline(V_src, L_src, nv_src, M)
        V_tgt_r, L_tgt_r, nv_tgt_r = resample_uniform_polyline(V_tgt, L_tgt, nv_tgt, M)
        V_final = scenario.run(
            (V_src_r, L_src_r, nv_src_r),
            (V_tgt_r, L_tgt_r, nv_tgt_r),
            matcher,
        )
        with torch.no_grad():
            out = chamfer((V_final, L_src_r, nv_src_r),
                          (V_tgt_r, L_tgt_r, nv_tgt_r))
            rewards.append(-out["chamfer_sym"])
    return torch.cat(rewards) if rewards else torch.empty(0)


def _evaluate(feature_extractor, eval_cfg, ds_src, ds_tgt, scenario, chamfer, M, device,
              compare_to_knn: bool, eval_batch_size: int):
    """Argmax rollout on a held-out subset; optionally also Knn3dMatcher rollout."""
    eval_src = _build_eval_subset(ds_src, eval_cfg.get("num_samples"),
                                  eval_cfg.get("seed_src", 100))
    eval_tgt = _build_eval_subset(ds_tgt, eval_cfg.get("num_samples"),
                                  eval_cfg.get("seed_tgt", 101))
    loader_src = DataLoader(eval_src, batch_size=eval_batch_size, shuffle=False,
                            num_workers=0, drop_last=False, collate_fn=shape_collate_fn)
    loader_tgt = DataLoader(eval_tgt, batch_size=eval_batch_size, shuffle=False,
                            num_workers=0, drop_last=False, collate_fn=shape_collate_fn)
    batches_src = list(loader_src)
    batches_tgt = list(loader_tgt)

    feature_extractor.eval()
    learned = LearnedMatcher(feature_extractor)
    R_learned = _rollout_reward(learned, scenario, chamfer, batches_src, batches_tgt, M, device)
    out = {
        "learned_reward_mean": float(R_learned.mean().item()),
        "learned_reward_std": float(R_learned.std().item() if R_learned.numel() > 1 else 0.0),
    }
    if compare_to_knn:
        knn = Knn3dMatcher(bidirectional=False)
        R_knn = _rollout_reward(knn, scenario, chamfer, batches_src, batches_tgt, M, device)
        out["knn_reward_mean"] = float(R_knn.mean().item())
        out["knn_reward_std"] = float(R_knn.std().item() if R_knn.numel() > 1 else 0.0)
    else:
        out["knn_reward_mean"] = float("nan")
        out["knn_reward_std"] = float("nan")
    feature_extractor.train()
    return out


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
    eval_log_path = os.path.join(output_dir, "eval_log.csv")
    ckpt_path = os.path.join(output_dir, "matcher.pt")
    OmegaConf.save(cfg, os.path.join(output_dir, "resolved_config.yaml"), resolve=True)

    eval_cfg = cfg.get("eval", None) or {}
    eval_every = eval_cfg.get("every_steps", None)
    eval_every = int(eval_every) if eval_every else None
    eval_batch_size = int(eval_cfg.get("batch_size", cfg.batch_size))
    eval_compare_to_knn = bool(eval_cfg.get("compare_to_knn", True))

    eval_logf = open(eval_log_path, "w") if eval_every is not None else None
    if eval_logf is not None:
        eval_logf.write(
            "step,learned_reward_mean,learned_reward_std,"
            "knn_reward_mean,knn_reward_std\n"
        )

    def _run_eval(step):
        if eval_logf is None:
            return
        metrics = _evaluate(
            matcher.feature_extractor, eval_cfg, ds_src, ds_tgt, scenario, chamfer,
            int(cfg.M), device, eval_compare_to_knn, eval_batch_size,
        )
        eval_logf.write(
            f"{step},{metrics['learned_reward_mean']:.6g},"
            f"{metrics['learned_reward_std']:.6g},"
            f"{metrics['knn_reward_mean']:.6g},"
            f"{metrics['knn_reward_std']:.6g}\n"
        )
        eval_logf.flush()
        tqdm.write(
            f"[eval @ step {step}] learned={metrics['learned_reward_mean']:.4f}"
            + (f"  knn={metrics['knn_reward_mean']:.4f}" if eval_compare_to_knn else "")
        )

    def _save_checkpoint(step):
        torch.save({
            "feature_extractor_state_dict": matcher.feature_extractor.state_dict(),
            "temperature": matcher.temperature,
            "M": int(cfg.M),
            "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, ckpt_path)

    num_steps = int(cfg.num_steps)
    checkpoint_every = cfg.get("checkpoint_every_steps", None)
    checkpoint_every = int(checkpoint_every) if checkpoint_every else None

    iter_src = _cycle(loader_src)
    iter_tgt = _cycle(loader_tgt)

    _run_eval(0)

    with open(log_path, "w") as logf:
        logf.write("step,reward_mean,reward_std,loss,entropy,baseline\n")

        pbar = tqdm(range(1, num_steps + 1), desc="train")
        for step in pbar:
            batch_src = next(iter_src)
            batch_tgt = next(iter_tgt)
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

            logf.write(
                f"{step},"
                f"{R.mean().item():.6g},{R.std().item():.6g},"
                f"{loss.item():.6g},{entropy.mean().item():.6g},"
                f"{b[0].item():.6g}\n"
            )
            logf.flush()

            pbar.set_postfix({
                "R": f"{R.mean().item():.4f}",
                "ent": f"{entropy.mean().item():.3f}",
                "loss": f"{loss.item():.4f}",
            })

            if checkpoint_every is not None and step % checkpoint_every == 0:
                _save_checkpoint(step)

            if eval_every is not None and step % eval_every == 0:
                _run_eval(step)

        _save_checkpoint(num_steps)
        if eval_every is not None and num_steps % eval_every != 0:
            _run_eval(num_steps)

    if eval_logf is not None:
        eval_logf.close()

    return ckpt_path


@hydra.main(version_base=None, config_path="../configs", config_name="train_matcher")
def main(cfg: DictConfig) -> None:
    out = train(cfg)
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()
