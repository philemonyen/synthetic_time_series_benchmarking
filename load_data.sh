#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <dataset>"
  echo "Available datasets: ptb-xl"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

DATASET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

case "${DATASET}" in
  ptb-xl)
    DATASET_DIR="${DATA_DIR}/ptb-xl"
    DATASET_URL="https://physionet.org/files/ptb-xl/1.0.3/"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    usage
    ;;
esac

mkdir -p "${DATASET_DIR}"

wget -r -N -c -np -nH --cut-dirs=3 -P "${DATASET_DIR}" "${DATASET_URL}"