"""Smoke test: classify circle / hexagon / triangle / 5-star polygons with PolygonCNN.

Run from repo root:
    python scripts/smoke_classify.py

Expectation: val accuracy should reach >>25% (random baseline for 4 classes) within
a few hundred steps. On a typical run a tiny CNN gets >95% — if it doesn't, the
encoder is probably miswired.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_generation.generate import ShapeGenerator  # noqa: E402
from rlmd.models import PolygonCNN  # noqa: E402


def normalize_polygons(V, num_verts):
    """Center each polygon at its centroid and rescale to unit mean radius.

    Removes the affine nuisance (translation + global scale) so the CNN can
    focus on shape rather than coordinate magnitude. Anisotropic scale and
    rotation are left in — they are real shape variation the network must
    learn to be invariant to.
    """
    N_max = V.shape[1]
    mask = (torch.arange(N_max, device=V.device)[None, :] < num_verts[:, None]).to(V.dtype)
    m = mask.unsqueeze(-1)
    n = num_verts.to(V.dtype)[:, None]
    centroid = (V * m).sum(dim=1) / n
    Vc = (V - centroid.unsqueeze(1)) * m
    mean_r = (Vc.norm(dim=-1) * mask).sum(dim=1) / n.squeeze(-1)
    Vc = Vc / mean_r.clamp(min=1e-6).view(-1, 1, 1)
    return Vc * m


def generate_dataset(n_per_class, num_points, seed, device):
    shapes = {
        "circle": {"shape": "circle", "num_points": num_points},
        "hexagon": {"shape": "hexagon", "num_points": num_points},
        "triangle": {"shape": "triangle", "num_points": num_points},
        "star_5": {"shape": "star", "num_points": num_points, "n_tips": 5, "inner_radius": 0.45},
    }
    transform_cfg = {
        "translation_range": [-1.0, 1.0],
        "scale_range": [0.5, 2.0],
        "rotation_range": [0.0, 360.0],
    }
    mixture = ShapeGenerator().generate_mixture_batch_torch(
        shapes=shapes,
        transform_cfg=transform_cfg,
        samples_per_shape=n_per_class,
        seed=seed,
        device=device,
    )

    Vs, Ls, ys, names = [], [], [], []
    for cls_idx, (name, batch) in enumerate(mixture.items()):
        pts = batch.points()                              # (B, P, 2)
        edges = batch.edges                               # (P, 2)
        B = pts.shape[0]
        Vs.append(pts)
        Ls.append(edges.unsqueeze(0).expand(B, -1, -1))
        ys.append(torch.full((B,), cls_idx, dtype=torch.long, device=device))
        names.append(name)

    V = torch.cat(Vs, dim=0)
    L = torch.cat(Ls, dim=0)
    y = torch.cat(ys, dim=0)
    num_verts = torch.full((V.shape[0],), V.shape[1], dtype=torch.long, device=device)
    return V, L, num_verts, y, names


class PolygonClassifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, V, L, num_verts, check_l=False):
        feats = self.encoder(V, L, num_verts, check_l=check_l)  # (B, N_max, F)
        N_max = feats.shape[1]
        mask = (torch.arange(N_max, device=V.device)[None, :] < num_verts[:, None])
        feats = feats.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = feats.max(dim=1).values
        return self.head(pooled)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-per-class", type=int, default=2000)
    p.add_argument("--num-points", type=int, default=60)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--no-normalize", action="store_true",
                   help="skip centroid/mean-radius normalization (sanity check)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    print(f"Generating data on {args.device}...")
    V, L, num_verts, y, class_names = generate_dataset(
        n_per_class=args.n_per_class,
        num_points=args.num_points,
        seed=args.seed,
        device=args.device,
    )
    if not args.no_normalize:
        V = normalize_polygons(V, num_verts)
    num_classes = len(class_names)
    print(f"  {V.shape[0]} samples, {num_classes} classes: {class_names}  "
          f"(normalize={'off' if args.no_normalize else 'on'})")

    perm = torch.randperm(V.shape[0], device=args.device)
    n_val = V.shape[0] // 5
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    feat_dim = 128
    encoder = PolygonCNN(
        in_channels=2,
        hidden_channels=(32, 64),
        out_channels=feat_dim,
        kernel_size=args.kernel_size,
        layernorm=True,
    )
    model = PolygonClassifier(encoder, feature_dim=feat_dim, num_classes=num_classes).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # One-time sanity check on the L tensor.
    _ = model(V[:2], L[:2], num_verts[:2], check_l=True)

    print(f"Training {args.steps} steps, batch size {args.batch_size}...")
    log_every = max(1, args.steps // 10)
    model.train()
    for step in range(args.steps):
        bi = train_idx[torch.randint(len(train_idx), (args.batch_size,), device=args.device)]
        logits = model(V[bi], L[bi], num_verts[bi])
        loss = F.cross_entropy(logits, y[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == args.steps - 1:
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
