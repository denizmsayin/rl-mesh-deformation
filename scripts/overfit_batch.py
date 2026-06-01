"""Single-fixed-batch overfit probe for the learned matcher.

Throwaway diagnostic (no Hydra): draws ONE batch from the online samplers,
resamples it once to M vertices, then repeatedly runs the exact training update
(StochasticLearnedMatcher -> SgdFixedMatchScenario -> Chamfer reward ->
REINFORCE) on that frozen batch. Mirrors scripts/train_matcher.py's per-step
math but never refreshes the data, so it isolates "can this encoder represent a
sharp matching at all?" from generalization / exploration.

Interpretation:
  - If entropy collapses and reward -> ~0, the encoder CAN fit one batch; the
    train/eval gap is optimization / exploration / generalization.
  - If entropy stays high (near ln(M)) and reward stalls, it's a
    representational symmetry problem (see the note in rlmd/models/polygon_gnn.py)
    -> positional encodings / global context / attention, not more tuning.

Run the SAME flags for --model cnn and --model gnn to compare on one batch:

    pixi run python scripts/overfit_batch.py --model cnn --steps 400
    pixi run python scripts/overfit_batch.py --model gnn --steps 400
"""
import argparse
import math

import torch

from rlmd.data.online import OnlineShapeSampler
from rlmd.evaluation.matchers import StochasticLearnedMatcher
from rlmd.evaluation.scenarios import SgdFixedMatchScenario
from rlmd.models import PolygonCNN, PolygonGNN
from rlmd.ops import resample_uniform_graph
from rlmd.training import BanditObjective, CompositeChamferReward, ScalarBaseline


# Target shape specs, mirroring configs/dataset_tgt/shapes_stream.yaml.
TGT_SPECS = {
    "circle": {"shape": "circle", "num_points": 60},
    "hexagon": {"shape": "hexagon", "num_points": 60},
    "triangle": {"shape": "triangle", "num_points": 60},
    "star": {"shape": "star", "num_points": 60, "n_tips": 5,
             "inner_radius": 0.45, "name": "star_5"},
}

# Transforms, mirroring the *_stream.yaml configs the cnn_vs_gnn runs used.
SRC_TRANSFORM = {
    "translation_range": [-1.0, 1.0],
    "scale_range": [0.5, 2.0],
    "rotation_range": [0.0, 0.0],
    "isotropic_scale": True,
}
TGT_TRANSFORM = {
    "translation_range": [-1.0, 1.0],
    "scale_range": [0.2, 2.0],
    "rotation_range": [0.0, 360.0],
    "isotropic_scale": False,
}


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _build_model(name: str, hidden, aggr: str):
    if name == "cnn":
        return PolygonCNN(in_channels=2, hidden_channels=hidden,
                          out_channels=128, kernel_size=5, layernorm=True)
    if name == "gnn":
        return PolygonGNN(in_channels=2, hidden_channels=hidden,
                          out_channels=128, aggr=aggr, layernorm=True)
    raise ValueError(f"unknown model: {name!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "gnn"], default="gnn")
    p.add_argument("--tgt-shape", choices=list(TGT_SPECS), default="star")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam lr. Matches scripts/train_matcher.py; higher "
                        "values (e.g. 1e-3) make REINFORCE collapse entropy "
                        "too fast and lock onto a bad deterministic matching.")
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--baseline", choices=["none", "ema"], default="ema",
                   help="REINFORCE baseline. 'none' (advantage=R) only works "
                        "with a large batch for contrast; 'ema' centers the "
                        "reward across steps and is far more stable on a "
                        "fixed/small batch.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--hidden", type=str, default="64,64,128",
                   help="comma-separated hidden channel widths")
    p.add_argument("--aggr", type=str, default="max",
                   help="GNN aggregation (max/mean/add); ignored for cnn")
    p.add_argument("--num-iters", type=int, default=400,
                   help="inner SGD iterations in the scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    if args.batch_size == 1 and args.baseline == "none":
        print("WARNING: batch_size=1 with baseline=none is a degenerate "
              "REINFORCE estimator (advantage=R is always negative, so every "
              "sampled action is suppressed without contrast). The policy will "
              "collapse to a confident-but-arbitrary matching and reward may "
              "WORSEN. Use a larger --batch-size and/or --baseline ema for a "
              "meaningful overfit test.")

    # One fixed batch: draw once, resample once, reuse every step.
    src_sampler = OnlineShapeSampler(
        shape_specs=[{"shape": "circle", "num_points": 60}],
        transform=SRC_TRANSFORM, seed=0,
    )
    tgt_sampler = OnlineShapeSampler(
        shape_specs=[TGT_SPECS[args.tgt_shape]],
        transform=TGT_TRANSFORM, seed=1,
    )

    V_src, L_src, nv_src, ne_src, _ = src_sampler.next_batch(args.batch_size, device)
    V_tgt, L_tgt, nv_tgt, ne_tgt, _ = tgt_sampler.next_batch(args.batch_size, device)

    src = resample_uniform_graph(V_src, L_src, nv_src, ne_src, args.M)
    tgt = resample_uniform_graph(V_tgt, L_tgt, nv_tgt, ne_tgt, args.M)

    model = _build_model(args.model, hidden, args.aggr).to(device)
    model.train()
    matcher = StochasticLearnedMatcher(model, temperature=args.temperature)

    scenario = SgdFixedMatchScenario(
        num_iters=args.num_iters, lr=0.0625, momentum=0.9,
        w_data=1.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1, distance_p=2,
    )
    reward = CompositeChamferReward(num_samples=8192, w_chamfer=1.0, w_normal=0.0)
    baseline = ScalarBaseline(kind=args.baseline)
    objective = BanditObjective(scenario, reward, baseline,
                                entropy_coef=args.entropy_coef)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} hidden={hidden} aggr="
          f"{args.aggr if args.model == 'gnn' else 'n/a'} params={n_params:,}")
    print(f"circle -> {args.tgt_shape}  batch={args.batch_size}  M={args.M}  "
          f"lr={args.lr}  entropy_coef={args.entropy_coef}  "
          f"baseline={args.baseline}  ln(M)={math.log(args.M):.3f}")
    print(f"{'step':>5} {'reward':>10} {'chamfer':>10} {'entropy':>9} "
          f"{'eff_tgts':>9} {'loss':>10}")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        update = objective.compute(matcher, src, tgt)
        update.loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            m = update.metrics
            eff = math.exp(m["entropy"])
            print(f"{step:>5} {m['reward_mean']:>10.5f} "
                  f"{m['reward_chamfer_mean']:>10.5f} {m['entropy']:>9.4f} "
                  f"{eff:>9.2f} {m['loss']:>10.4f}")


if __name__ == "__main__":
    main()
