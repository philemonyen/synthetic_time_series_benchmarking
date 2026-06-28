#!/usr/bin/env bash
#SBATCH --account=def-chenh
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=sssd_benchmark
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <model>"
  echo "  model: model name (e.g. sssd-ecg)"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

MODEL="$1"

# 1. Load environment modules
module --force purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load scipy-stack

# 2. Export CUDA paths
export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${EBROOTCUDA}/bin:${PATH}"

./generate.sh "${MODEL}"
