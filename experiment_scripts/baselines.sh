#! /bin/bash

for b in none ema prior chamfer_sgd; do
  python scripts/train_matcher.py total_trajectories=5000000 eval.every_steps=100 src=circle_translated tgt=shapes M=128 scenario.num_iters=400 reward.w_normal=0.0 baseline=$b
done
