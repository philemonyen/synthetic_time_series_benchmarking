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
  echo "Usage: $(basename "$0") <dataset> <model>"
  echo "  dataset: dataset name (e.g. ptb-xl), or '-' to skip data download"
  echo "  model:   model name (e.g. sssd-ecg)"
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
fi

DATASET="$1"
MODEL="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

# 3. Optional dataset download
if [[ "${DATASET}" != "-" && -n "${DATASET}" ]]; then
  echo "Downloading dataset: ${DATASET}"
  ./load_data.sh "${DATASET}"
fi

# 4. Download model and create virtual environment
echo "Loading model: ${MODEL}"
./load_model.sh "${MODEL}"

# 5. Activate model virtual environment
case "${MODEL}" in
  sssd-ecg)
    VENV_DIR="${SCRIPT_DIR}/model/SSSD-ECG/venv"
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Error: Virtual environment not found at ${VENV_DIR}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# 6. Train and generate synthetic data
./generate.sh "${MODEL}"
