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
import copy
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rlmd.dataset import shape_collate_fn
from rlmd.evaluation.matchers import (
    FixedMatcher,
    Knn3dMatcher,
    LearnedMatcher,
    StochasticLearnedMatcher,
)


class _StochasticRolloutMatcher:
    """Adapter so StochasticLearnedMatcher conforms to the deterministic
    Matcher protocol (returns List[Matching] only). For sampled-eval rollouts
    where log_prob and entropy are not needed."""

    name = "stochastic_rollout"

    def __init__(self, base: StochasticLearnedMatcher):
        self._base = base

    def __call__(self, V_src, n_src, V_tgt, n_tgt):
        matchings, _lp, _ent = self._base(V_src, n_src, V_tgt, n_tgt)
        return matchings
from rlmd.evaluation.metrics import ChamferMetric
from rlmd.ops import resample_uniform_polyline


def _resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _to_device(batch, device):
    V, L, lengths, shapes = batch
    return V.to(device), L.to(device), lengths.to(device), shapes


def _build_eval_subset(dataset, num_samples, seed):
    """Deterministic Subset of `dataset` with `num_samples` items (or fewer)."""
    if num_samples is None or num_samples >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=g)[:int(num_samples)].tolist()
    return Subset(dataset, indices)


def _compute_reward(out, w_chamfer, w_normal):
    """Composite per-sample reward from a ChamferMetric(with_normals=True) output."""
    return -(w_chamfer * out["chamfer_sym"] + w_normal * out["normal_sym"])


def _rollout_reward(matcher, scenario, chamfer, batches_src, batches_tgt, M, device,
                    w_chamfer, w_normal):
    """Run scenario for each pair, return concatenated per-item composite rewards.

    NOTE: cannot wrap the scenario call in torch.no_grad() because the scenario's
    inner SGD requires gradients on `deform`. Both LearnedMatcher and
    Knn3dMatcher avoid leaking gradients to feature-extractor params (the
    former wraps its forward in no_grad, the latter has no params). Only the
    reward Chamfer is wrapped in no_grad here.
    """
    assert len(batches_src) == len(batches_tgt), (
        f"src/tgt batch counts disagree: {len(batches_src)} vs {len(batches_tgt)}"
    )
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
            rewards.append(_compute_reward(out, w_chamfer, w_normal))
    return torch.cat(rewards) if rewards else torch.empty(0)


def _evaluate(feature_extractor, eval_cfg, ds_src, ds_tgt, scenario, chamfer, M, device,
              compare_to_knn: bool, eval_batch_size: int, w_chamfer: float, w_normal: float,
              streamed_batches=None, prior_matcher=None):
    """Argmax rollout on a held-out subset; optionally also Knn3dMatcher rollout.

    If ``streamed_batches`` is given as a (list[batch_src], list[batch_tgt]) pair,
    also run learned + prior rollouts on it for an in-training-distribution
    diagnostic. If ``prior_matcher`` is also given, its rollout is reported.
    """
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
    R_learned = _rollout_reward(learned, scenario, chamfer, batches_src, batches_tgt, M, device,
                                w_chamfer, w_normal)
    out = {
        "learned_reward_mean": float(R_learned.mean().item()),
        "learned_reward_std": float(R_learned.std().item() if R_learned.numel() > 1 else 0.0),
    }
    if streamed_batches is not None:
        s_src, s_tgt = streamed_batches
        R_stream_learned = _rollout_reward(learned, scenario, chamfer, s_src, s_tgt, M, device,
                                           w_chamfer, w_normal)
        out["stream_learned_reward_mean"] = float(R_stream_learned.mean().item())
        out["stream_learned_reward_std"] = float(
            R_stream_learned.std().item() if R_stream_learned.numel() > 1 else 0.0)

        # Sampled rollout from the same policy: matches the distribution that
        # REINFORCE actually optimizes (E_{c~π_θ}[R(c)]). Averaging over
        # `sampled_num_rollouts` independent samples per state reduces noise.
        n_samp = int(eval_cfg.get("sampled_num_rollouts", 1))
        stoch = _StochasticRolloutMatcher(StochasticLearnedMatcher(feature_extractor))
        n_samp = max(1, n_samp)
        R_acc = _rollout_reward(stoch, scenario, chamfer, s_src, s_tgt, M, device,
                                w_chamfer, w_normal)
        for _ in range(n_samp - 1):
            R_acc = R_acc + _rollout_reward(stoch, scenario, chamfer, s_src, s_tgt, M,
                                            device, w_chamfer, w_normal)
        R_stream_sampled = R_acc / n_samp
        out["stream_sampled_reward_mean"] = float(R_stream_sampled.mean().item())
        out["stream_sampled_reward_std"] = float(
            R_stream_sampled.std().item() if R_stream_sampled.numel() > 1 else 0.0)

        if prior_matcher is not None:
            R_stream_prior = _rollout_reward(prior_matcher, scenario, chamfer, s_src, s_tgt, M,
                                             device, w_chamfer, w_normal)
            out["stream_prior_reward_mean"] = float(R_stream_prior.mean().item())
            out["stream_prior_reward_std"] = float(
                R_stream_prior.std().item() if R_stream_prior.numel() > 1 else 0.0)
        else:
            out["stream_prior_reward_mean"] = float("nan")
            out["stream_prior_reward_std"] = float("nan")
    else:
        out["stream_learned_reward_mean"] = float("nan")
        out["stream_learned_reward_std"] = float("nan")
        out["stream_sampled_reward_mean"] = float("nan")
        out["stream_sampled_reward_std"] = float("nan")
        out["stream_prior_reward_mean"] = float("nan")
        out["stream_prior_reward_std"] = float("nan")
    if compare_to_knn:
        knn = Knn3dMatcher(bidirectional=False)
        R_knn = _rollout_reward(knn, scenario, chamfer, batches_src, batches_tgt, M, device,
                                w_chamfer, w_normal)
        out["knn_reward_mean"] = float(R_knn.mean().item())
        out["knn_reward_std"] = float(R_knn.std().item() if R_knn.numel() > 1 else 0.0)

        knn_bi = Knn3dMatcher(bidirectional=True)
        R_knn_bi = _rollout_reward(knn_bi, scenario, chamfer, batches_src, batches_tgt, M,
                                   device, w_chamfer, w_normal)
        out["knn_bi_reward_mean"] = float(R_knn_bi.mean().item())
        out["knn_bi_reward_std"] = float(R_knn_bi.std().item() if R_knn_bi.numel() > 1 else 0.0)
    else:
        out["knn_reward_mean"] = float("nan")
        out["knn_reward_std"] = float("nan")
        out["knn_bi_reward_mean"] = float("nan")
        out["knn_bi_reward_std"] = float("nan")
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
        # ema: return the current (pre-update) value, then update with this batch.
        # First-batch fallback uses the batch mean so the very first advantage is
        # centered rather than equal to R.
        batch_mean = R.detach().mean().item()
        b = batch_mean if self.value is None else self.value
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * batch_mean
        return torch.full_like(R, b)


def _save_training_curves(log_path: str, out_path: str, baseline_type: str) -> None:
    """Plot reward / baseline / advantage from the per-step CSV log."""
    import csv

    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    step = np.array([int(r["step"]) for r in rows])
    R = np.array([float(r["reward_mean"]) for r in rows])
    b = np.array([float(r["baseline"]) for r in rows])
    adv = np.array([float(r["advantage_mean"]) for r in rows])
    R_argmax = None
    if "argmax_reward_mean" in rows[0]:
        try:
            R_argmax = np.array([float(r["argmax_reward_mean"]) for r in rows])
        except (KeyError, ValueError):
            R_argmax = None

    def _smooth(x):
        w = max(1, min(200, len(x) // 20))
        if w <= 1:
            return x, np.arange(len(x))
        k = np.ones(w) / w
        s = np.convolve(x, k, mode="valid")
        off = (w - 1) // 2
        return s, np.arange(off, off + len(s))

    # Optional: read eval_log.csv from the same directory and overlay the
    # streamed-eval signal (same distribution as training, fixed seed) on the
    # top panel. Lives on the same y-scale as the training reward so direct
    # overlay makes sense.
    eval_log_path = os.path.join(os.path.dirname(log_path), "eval_log.csv")
    eval_step = eval_stream_learned = eval_stream_prior = eval_stream_sampled = None
    if os.path.exists(eval_log_path):
        e_rows = []
        with open(eval_log_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                e_rows.append(r)
        if e_rows and "stream_learned_reward_mean" in e_rows[0]:
            def _maybe_float(s):
                try:
                    return float(s)
                except (TypeError, ValueError):
                    return float("nan")
            eval_step = np.array([int(r["step"]) for r in e_rows])
            eval_stream_learned = np.array(
                [_maybe_float(r["stream_learned_reward_mean"]) for r in e_rows])
            eval_stream_prior = np.array(
                [_maybe_float(r["stream_prior_reward_mean"]) for r in e_rows])
            if "stream_sampled_reward_mean" in e_rows[0]:
                eval_stream_sampled = np.array(
                    [_maybe_float(r["stream_sampled_reward_mean"]) for r in e_rows])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(step, R, color="C0", alpha=0.25, lw=0.6,
                 label="reward sampled (raw)")
    axes[0].plot(step, b, color="C1", alpha=0.25, lw=0.6,
                 label=f"baseline ({baseline_type}) (raw)")
    R_s, R_x = _smooth(R)
    b_s, b_x = _smooth(b)
    axes[0].plot(step[R_x], R_s, color="C0", lw=1.5,
                 label="reward sampled (smoothed)")
    axes[0].plot(step[b_x], b_s, color="C1", lw=1.5, label="baseline (smoothed)")
    if R_argmax is not None:
        axes[0].plot(step, R_argmax, color="C4", alpha=0.25, lw=0.6,
                     label="reward argmax (raw)")
        Ra_s, Ra_x = _smooth(R_argmax)
        axes[0].plot(step[Ra_x], Ra_s, color="C4", lw=1.5,
                     label="reward argmax (smoothed)")
    if eval_step is not None and eval_stream_learned is not None and np.isfinite(
            eval_stream_learned).any():
        axes[0].plot(eval_step, eval_stream_learned, "o-", color="C0",
                     markersize=4, lw=0.8, label="stream eval learned (argmax)")
        if eval_stream_sampled is not None and np.isfinite(eval_stream_sampled).any():
            axes[0].plot(eval_step, eval_stream_sampled, "s--", color="C4",
                         markersize=4, lw=0.8, label="stream eval learned (sampled)")
        if eval_stream_prior is not None and np.isfinite(eval_stream_prior).any():
            axes[0].plot(eval_step, eval_stream_prior, "o-", color="C1",
                         markersize=4, lw=0.8, label="stream eval prior")
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].axhline(0.0, color="k", lw=0.6, alpha=0.5)
    axes[1].plot(step, adv, color="C2", alpha=0.25, lw=0.6,
                 label="advantage sampled (raw)")
    adv_s, adv_x = _smooth(adv)
    axes[1].plot(step[adv_x], adv_s, color="C2", lw=1.5,
                 label="advantage sampled (smoothed)")
    if R_argmax is not None:
        adv_argmax = R_argmax - b
        axes[1].plot(step, adv_argmax, color="C4", alpha=0.25, lw=0.6,
                     label="advantage argmax (raw)")
        aa_s, aa_x = _smooth(adv_argmax)
        axes[1].plot(step[aa_x], aa_s, color="C4", lw=1.5,
                     label="advantage argmax (smoothed)")
    if (eval_step is not None and eval_stream_learned is not None
            and eval_stream_prior is not None
            and np.isfinite(eval_stream_learned).any()
            and np.isfinite(eval_stream_prior).any()):
        axes[1].plot(eval_step, eval_stream_learned - eval_stream_prior, "o-",
                     color="C3", markersize=4, lw=0.8,
                     label="stream eval advantage (argmax)")
        if eval_stream_sampled is not None and np.isfinite(eval_stream_sampled).any():
            axes[1].plot(eval_step, eval_stream_sampled - eval_stream_prior, "s--",
                         color="C4", markersize=4, lw=0.8,
                         label="stream eval advantage (sampled)")
    axes[1].set_ylabel("advantage (R - baseline)")
    axes[1].set_xlabel("step")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Training curves (baseline={baseline_type})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def train(cfg: DictConfig) -> str:
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    src_source = instantiate(cfg.dataset_src.source)
    tgt_source = instantiate(cfg.dataset_tgt.source)

    matcher = instantiate(cfg.matcher)
    matcher.feature_extractor.to(device)
    matcher.feature_extractor.train()

    # Keep an uncompiled handle for state_dict saves and optimizer params; the
    # matcher's `feature_extractor` attribute is swapped for the compiled
    # wrapper so all forward calls (training + argmax eval) go through it.
    feature_extractor = matcher.feature_extractor

    # SCST-style baseline: snapshot the initial extractor (frozen, deterministic
    # argmax) as a state-conditional reference. Must be deep-copied BEFORE
    # torch.compile so the prior is a clean uncompiled module that doesn't
    # share weight tensors with the trainable matcher.
    baseline_type = str(cfg.baseline.get("type", "none"))
    ema_baseline = None
    prior_matcher = None
    if baseline_type == "prior":
        prior_extractor = copy.deepcopy(feature_extractor)
        for p in prior_extractor.parameters():
            p.requires_grad = False
        prior_matcher = LearnedMatcher(torch.compile(prior_extractor),
                                       temperature=matcher.temperature)
    else:
        ema_baseline = _Baseline(cfg.baseline)

    matcher.feature_extractor = torch.compile(feature_extractor)

    # Argmax view of the *current* policy, for the inline diagnostic rollout
    # at each training step. Constructed once: LearnedMatcher's __init__ flips
    # the extractor to eval mode, so we restore train mode immediately. Future
    # __call__ invocations don't touch the mode.
    policy_argmax_matcher = LearnedMatcher(matcher.feature_extractor,
                                           temperature=matcher.temperature)
    feature_extractor.train()

    optimizer = instantiate(cfg.optimizer, params=feature_extractor.parameters())

    scenario = instantiate(cfg.scenario)

    chamfer = ChamferMetric(num_samples=int(cfg.reward_num_samples),
                            point_reduction="mean", norm=2, with_normals=True)

    w_chamfer = float(cfg.reward.w_chamfer)
    w_normal = float(cfg.reward.w_normal)

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
    streamed_eval_batches = None
    if eval_logf is not None:
        eval_logf.write(
            "step,learned_reward_mean,learned_reward_std,"
            "knn_reward_mean,knn_reward_std,"
            "knn_bi_reward_mean,knn_bi_reward_std,"
            "stream_learned_reward_mean,stream_learned_reward_std,"
            "stream_sampled_reward_mean,stream_sampled_reward_std,"
            "stream_prior_reward_mean,stream_prior_reward_std\n"
        )
        eval_ds_src = instantiate(cfg.eval.dataset_src.dataset)
        eval_ds_tgt = instantiate(cfg.eval.dataset_tgt.dataset)

        # Streamed-eval batch: same distribution as training, but reproducible
        # across evals. Built once at startup with eval-specific seeds drawn
        # from `eval.stream_seed_src/tgt`, falling back to seed_src/seed_tgt
        # offset by 1000 to avoid colliding with the disk-eval seeds.
        eval_stream_n = int(eval_cfg.get("num_samples", 64))
        stream_seed_src = int(eval_cfg.get("stream_seed_src",
                                           int(eval_cfg.get("seed_src", 100)) + 1000))
        stream_seed_tgt = int(eval_cfg.get("stream_seed_tgt",
                                           int(eval_cfg.get("seed_tgt", 101)) + 1000))
        src_eval_sampler = instantiate(cfg.dataset_src.source, seed=stream_seed_src)
        tgt_eval_sampler = instantiate(cfg.dataset_tgt.source, seed=stream_seed_tgt)
        cpu = torch.device("cpu")
        streamed_eval_batches = (
            [src_eval_sampler.next_batch(eval_stream_n, cpu)],
            [tgt_eval_sampler.next_batch(eval_stream_n, cpu)],
        )
    else:
        eval_ds_src = eval_ds_tgt = None

    def _run_eval(step):
        if eval_logf is None:
            return
        metrics = _evaluate(
            matcher.feature_extractor, eval_cfg, eval_ds_src, eval_ds_tgt,
            scenario, chamfer, int(cfg.M), device,
            eval_compare_to_knn, eval_batch_size,
            w_chamfer, w_normal,
            streamed_batches=streamed_eval_batches,
            prior_matcher=prior_matcher,
        )
        eval_logf.write(
            f"{step},{metrics['learned_reward_mean']:.6g},"
            f"{metrics['learned_reward_std']:.6g},"
            f"{metrics['knn_reward_mean']:.6g},"
            f"{metrics['knn_reward_std']:.6g},"
            f"{metrics['knn_bi_reward_mean']:.6g},"
            f"{metrics['knn_bi_reward_std']:.6g},"
            f"{metrics['stream_learned_reward_mean']:.6g},"
            f"{metrics['stream_learned_reward_std']:.6g},"
            f"{metrics['stream_sampled_reward_mean']:.6g},"
            f"{metrics['stream_sampled_reward_std']:.6g},"
            f"{metrics['stream_prior_reward_mean']:.6g},"
            f"{metrics['stream_prior_reward_std']:.6g}\n"
        )
        eval_logf.flush()
        msg = f"[eval @ step {step}] learned={metrics['learned_reward_mean']:.4f}"
        if eval_compare_to_knn:
            msg += (f"  knn={metrics['knn_reward_mean']:.4f}"
                    f"  knn_bi={metrics['knn_bi_reward_mean']:.4f}")
        if streamed_eval_batches is not None:
            msg += (f"  stream={metrics['stream_learned_reward_mean']:.4f}"
                    f"  stream_samp={metrics['stream_sampled_reward_mean']:.4f}")
            if prior_matcher is not None:
                msg += f"  stream_prior={metrics['stream_prior_reward_mean']:.4f}"
        tqdm.write(msg)

    def _save_checkpoint(step):
        torch.save({
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "temperature": matcher.temperature,
            "M": int(cfg.M),
            "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, ckpt_path)

    total_trajectories = int(cfg.total_trajectories)

    checkpoint_every = cfg.get("checkpoint_every_steps", None)
    checkpoint_every = int(checkpoint_every) if checkpoint_every else None

    batch_size = int(cfg.batch_size)

    _run_eval(0)

    with open(log_path, "w") as logf:
        logf.write(
            "step,traj,reward_mean,reward_std,"
            "reward_chamfer_mean,reward_normal_mean,"
            "advantage_mean,advantage_std,"
            "argmax_reward_mean,argmax_reward_std,"
            "loss,entropy,baseline\n"
        )

        pbar = tqdm(total=total_trajectories, desc="train", unit="traj")
        step = 0
        traj = 0
        while traj < total_trajectories:
            step += 1
            V_src, L_src, nv_src, _ = src_source.next_batch(batch_size, device)
            V_tgt, L_tgt, nv_tgt, _ = tgt_source.next_batch(batch_size, device)
            B = V_src.shape[0]
            traj += B

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
                R = _compute_reward(out, w_chamfer, w_normal)  # (B,)

            # Diagnostic: rollout the policy's *argmax* matching on the same
            # batch. Not used for the loss — just a logged signal so we can
            # compare argmax-π_θ vs sampled-π_θ vs argmax-π_0 (baseline) on
            # the same batch. One extra scenario rollout per training step.
            with torch.no_grad():
                policy_argmax_matchings = policy_argmax_matcher(
                    V_src_r, nv_src_r, V_tgt_r, nv_tgt_r)
            V_final_argmax = scenario.run(
                (V_src_r, L_src_r, nv_src_r),
                (V_tgt_r, L_tgt_r, nv_tgt_r),
                FixedMatcher(policy_argmax_matchings),
            )
            with torch.no_grad():
                out_argmax = chamfer((V_final_argmax, L_src_r, nv_src_r),
                                     (V_tgt_r, L_tgt_r, nv_tgt_r))
                R_argmax = _compute_reward(out_argmax, w_chamfer, w_normal)  # (B,)

            # Baseline: either scalar EMA (or zero) or per-state SCST rollout
            # under the frozen prior policy. The prior rollout costs one extra
            # scenario evaluation per step.
            if prior_matcher is not None:
                prior_matchings = prior_matcher(V_src_r, nv_src_r, V_tgt_r, nv_tgt_r)
                V_final_prior = scenario.run(
                    (V_src_r, L_src_r, nv_src_r),
                    (V_tgt_r, L_tgt_r, nv_tgt_r),
                    FixedMatcher(prior_matchings),
                )
                with torch.no_grad():
                    out_prior = chamfer((V_final_prior, L_src_r, nv_src_r),
                                        (V_tgt_r, L_tgt_r, nv_tgt_r))
                    b = _compute_reward(out_prior, w_chamfer, w_normal)  # (B,)
            else:
                assert ema_baseline is not None
                with torch.no_grad():
                    b = ema_baseline(R)                                  # (B,)

            advantage = (R - b).detach()
            loss = -(advantage * log_prob).mean() \
                   - float(cfg.entropy_coef) * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_std = advantage.std().item() if advantage.numel() > 1 else 0.0
            argmax_std = (R_argmax.std().item()
                          if R_argmax.numel() > 1 else 0.0)
            logf.write(
                f"{step},{traj},"
                f"{R.mean().item():.6g},{R.std().item():.6g},"
                f"{out['chamfer_sym'].mean().item():.6g},"
                f"{out['normal_sym'].mean().item():.6g},"
                f"{advantage.mean().item():.6g},{adv_std:.6g},"
                f"{R_argmax.mean().item():.6g},{argmax_std:.6g},"
                f"{loss.item():.6g},{entropy.mean().item():.6g},"
                f"{b.mean().item():.6g}\n"
            )
            logf.flush()

            pbar.update(B)
            pbar.set_postfix({
                "R": f"{R.mean().item():.4f}",
                "ent": f"{entropy.mean().item():.3f}",
                "loss": f"{loss.item():.4f}",
            })

            if checkpoint_every is not None and step % checkpoint_every == 0:
                _save_checkpoint(step)

            if eval_every is not None and step % eval_every == 0:
                _run_eval(step)

        pbar.close()

        _save_checkpoint(step)
        if eval_every is not None and step % eval_every != 0:
            _run_eval(step)

    if eval_logf is not None:
        eval_logf.close()

    try:
        _save_training_curves(
            log_path,
            os.path.join(output_dir, "training_curves.png"),
            baseline_type,
        )
    except Exception as e:
        print(f"warning: failed to render training curves: {e}")

    return ckpt_path


@hydra.main(version_base=None, config_path="../configs", config_name="train_matcher")
def main(cfg: DictConfig) -> None:
    out = train(cfg)
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()
