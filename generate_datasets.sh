#!/usr/bin/env bash
# Regenerate every on-disk shape set referenced by configs/dataset_{src,tgt}/*_set.yaml.
# Edit N or add new lines as sets are added.
set -euo pipefail

cd "$(dirname "$0")"

N="${N:-2000}"

run() {
  echo "=== Generating $3 ==="
  pixi run python scripts/generate.py "dataset=$1" "dataset/transform=$2" "dataset_name=$3" "N=$N"
}

run circle_only   centered    circle_centered_set
run circle_only   scaled      circle_scaled_set
run circle_only   translated  circle_translated_set
run triangle_only centered    triangle_centered_set
run shapes_basic  default     shapes_set

echo
echo "All datasets generated."
