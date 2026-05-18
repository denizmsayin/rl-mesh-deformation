"""Smoke test: classify shapes with a configurable polygon encoder.

Run from repo root:
    python scripts/smoke_classify.py
    python scripts/smoke_classify.py train.steps=200 model.kernel_size=3
    python scripts/smoke_classify.py dataset/transform=default_no_translation normalize=false
    python scripts/smoke_classify.py model.hidden_channels=[64,128] samples_per_class=4000

Expectation: val accuracy >> 1/num_classes within a few hundred steps. On a typical
run the tiny PolygonCNN gets >95% — if it doesn't, the encoder is probably miswired.
"""
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from rlmd.data.generation import ShapeGenerator


def normalize_polygons(V, num_verts):
    """Center each polygon at its centroid and rescale to unit mean radius."""
    N_max = V.shape[1]
    mask = (torch.arange(N_max, device=V.device)[None, :] < num_verts[:, None]).to(V.dtype)
    m = mask.unsqueeze(-1)
    n = num_verts.to(V.dtype)[:, None]
    centroid = (V * m).sum(dim=1) / n
    Vc = (V - centroid.unsqueeze(1)) * m
    mean_r = (Vc.norm(dim=-1) * mask).sum(dim=1) / n.squeeze(-1)
    Vc = Vc / mean_r.clamp(min=1e-6).view(-1, 1, 1)
    return Vc * m


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
    y = torch.zeros(total_B, dtype=torch.long, device=device)
    names = []

    i = 0
    for cls_idx, (name, batch) in enumerate(items):
        pts = batch.points()
        edges = batch.edges
        B, P, _ = pts.shape
        V[i:i + B, :P] = pts
        L[i:i + B, :P] = edges.unsqueeze(0).expand(B, -1, -1)
        num_verts[i:i + B] = P
        y[i:i + B] = cls_idx
        names.append(name)
        i += B

    return V, L, num_verts, y, names


class PolygonClassifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, V, L, num_verts, check_l=False):
        feats = self.encoder(V, L, num_verts, check_l=check_l)
        N_max = feats.shape[1]
        mask = (torch.arange(N_max, device=V.device)[None, :] < num_verts[:, None])
        feats = feats.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return self.head(feats.max(dim=1).values)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    idx = y_true * num_classes + y_pred
    cm.view(-1).scatter_add_(0, idx, torch.ones_like(idx))
    return cm


def print_confusion(cm, names):
    width = max(len(n) for n in names) + 1
    header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "  ".join(f"{cm[i, j].item():>{width}d}" for j in range(len(names)))
        print(f"  {name:>{width}}  {row}")


def resolve_device(device_cfg):
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="classify")
def main(cfg: DictConfig):
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.train.seed)

    print(f"Generating data on {device}...")
    V, L, num_verts, y, class_names = generate_dataset(
        shapes=cfg.dataset.shapes,
        transform_cfg=cfg.dataset.transform,
        samples_per_class=cfg.samples_per_class,
        seed=cfg.train.seed,
        device=device,
    )
    if cfg.normalize:
        V = normalize_polygons(V, num_verts)
    num_classes = len(class_names)
    print(f"  {V.shape[0]} samples, {num_classes} classes: {class_names}  "
          f"(normalize={'on' if cfg.normalize else 'off'})")

    perm = torch.randperm(V.shape[0], device=device)
    n_val = int(V.shape[0] * cfg.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    encoder = instantiate(cfg.model)
    feat_dim = cfg.model.out_channels
    model = PolygonClassifier(encoder, feature_dim=feat_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    _ = model(V[:2], L[:2], num_verts[:2], check_l=True)

    print(f"Training {cfg.train.steps} steps, batch size {cfg.train.batch_size}...")
    log_every = max(1, cfg.train.steps // 10)
    model.train()
    for step in range(cfg.train.steps):
        bi = train_idx[torch.randint(len(train_idx), (cfg.train.batch_size,), device=device)]
        logits = model(V[bi], L[bi], num_verts[bi])
        loss = F.cross_entropy(logits, y[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == cfg.train.steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y[bi]).float().mean().item()
            print(f"  step {step:4d}  loss={loss.item():.4f}  train_acc(batch)={acc:.3f}")

    model.eval()
    with torch.no_grad():
        logits = model(V[val_idx], L[val_idx], num_verts[val_idx])
        pred = logits.argmax(-1)
        acc = (pred == y[val_idx]).float().mean().item()
        cm = confusion_matrix(y[val_idx].cpu(), pred.cpu(), num_classes)

    print(f"\nVal accuracy: {acc:.3f}  (random baseline: {1.0 / num_classes:.3f})")
    print("Confusion matrix (rows=true, cols=pred):")
    print_confusion(cm, class_names)


if __name__ == "__main__":
    main()
