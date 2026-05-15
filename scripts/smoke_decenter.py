"""Smoke test: regress the per-shape translation (learn to zero-center).

Run from repo root:
    python scripts/smoke_decenter.py
    python scripts/smoke_decenter.py model.layernorm=false
    python scripts/smoke_decenter.py train.steps=800 model.layernorm=false
    python scripts/smoke_decenter.py dataset.shapes='{circle:{shape:circle,num_points:60}}'

Expectation: val MSE should approach the analytical "predict vertex mean"
baseline (which is exact when base shapes are zero-centered and uniformly
sampled). If model MSE is much worse than that baseline, the encoder is
dropping absolute-position information — try `model.layernorm=false`, since
LayerNorm over the channel dim partially strips the translation signal that
the first conv just encoded.
"""
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_generation.generate import ShapeGenerator  # noqa: E402


def generate_dataset(shapes, transform_cfg, samples_per_class, seed, device):
    mixture = ShapeGenerator().generate_mixture_batch_torch(
        shapes=shapes,
        transform_cfg=transform_cfg,
        samples_per_shape=samples_per_class,
        seed=seed,
        device=device,
    )

    items = list(mixture.items())
    total_B = sum(batch.points().shape[0] for _, batch in items)
    max_P = max(batch.points().shape[1] for _, batch in items)

    V = torch.zeros(total_B, max_P, 2, device=device)
    L = torch.zeros(total_B, max_P, 2, dtype=torch.long, device=device)
    num_verts = torch.zeros(total_B, dtype=torch.long, device=device)
    t = torch.zeros(total_B, 2, device=device)
    names = []

    i = 0
    for name, batch in items:
        pts = batch.points()
        edges = batch.edges
        B, P, _ = pts.shape
        V[i:i + B, :P] = pts
        L[i:i + B, :P] = edges.unsqueeze(0).expand(B, -1, -1)
        num_verts[i:i + B] = P
        t[i:i + B] = batch.translation
        names.append(name)
        i += B

    return V, L, num_verts, t, names


def masked_mean(x, num_verts):
    """Mean over valid vertices. x: (B, N_max, C). Returns (B, C)."""
    N_max = x.shape[1]
    mask = (torch.arange(N_max, device=x.device)[None, :] < num_verts[:, None]).to(x.dtype)
    summed = (x * mask.unsqueeze(-1)).sum(dim=1)
    return summed / num_verts.to(x.dtype).clamp(min=1).unsqueeze(-1)


class PolygonRegressor(nn.Module):
    def __init__(self, encoder, feature_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, V, L, num_verts, check_l=False):
        feats = self.encoder(V, L, num_verts, check_l=check_l)
        return self.head(masked_mean(feats, num_verts))


def resolve_device(device_cfg):
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="decenter")
def main(cfg: DictConfig):
    device = resolve_device(cfg.device)
    is_cuda = torch.device(device).type == "cuda"
    torch.manual_seed(cfg.train.seed)

    print(f"Generating data on {device}...")
    V, L, num_verts, t, class_names = generate_dataset(
        shapes=cfg.dataset.shapes,
        transform_cfg=cfg.dataset.transform,
        samples_per_class=cfg.samples_per_class,
        seed=cfg.train.seed,
        device=device,
    )
    print(f"  {V.shape[0]} samples across {len(class_names)} shape(s): {class_names}")
    print(f"  encoder layernorm = {cfg.model.layernorm}")

    perm = torch.randperm(V.shape[0], device=device)
    n_val = int(V.shape[0] * cfg.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    encoder = instantiate(cfg.model)
    model = PolygonRegressor(encoder, feature_dim=cfg.model.out_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    _ = model(V[:2], L[:2], num_verts[:2], check_l=True)

    with torch.no_grad():
        vmean = masked_mean(V[val_idx], num_verts[val_idx])
        centroid_mse = F.mse_loss(vmean, t[val_idx]).item()
        zero_mse = F.mse_loss(torch.zeros_like(t[val_idx]), t[val_idx]).item()
    print(f"Baselines on val:  predict-zero MSE={zero_mse:.4e}   "
          f"predict-vertex-mean MSE={centroid_mse:.4e}")

    print(f"Training {cfg.train.steps} steps, batch size {cfg.train.batch_size}...")
    log_every = max(1, cfg.train.steps // 10)
    model.train()
    if is_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    t_last = t_start
    steps_since_log = 0
    for step in range(cfg.train.steps):
        bi = train_idx[torch.randint(len(train_idx), (cfg.train.batch_size,), device=device)]
        pred = model(V[bi], L[bi], num_verts[bi])
        loss = F.mse_loss(pred, t[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps_since_log += 1
        if step % log_every == 0 or step == cfg.train.steps - 1:
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            ms_per_step = (now - t_last) / max(steps_since_log, 1) * 1000.0
            elapsed = now - t_start
            print(f"  step {step:4d}  loss={loss.item():.4e}  "
                  f"{ms_per_step:6.2f} ms/step  elapsed={elapsed:6.2f}s")
            t_last = now
            steps_since_log = 0

    model.eval()
    with torch.no_grad():
        pred = model(V[val_idx], L[val_idx], num_verts[val_idx])
        mse = F.mse_loss(pred, t[val_idx]).item()
        l2 = (pred - t[val_idx]).norm(dim=-1).mean().item()

    ratio = mse / max(centroid_mse, 1e-12)
    print(f"\nVal MSE: {mse:.4e}   mean L2 error: {l2:.4e}")
    print(f"Model / vertex-mean-baseline MSE ratio: {ratio:.3f}  "
          "(>>1 ⇒ encoder is dropping absolute position; try model.layernorm=false)")


if __name__ == "__main__":
    main()
