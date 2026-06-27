#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <mode> <scp-destination>"
  echo "Modes:"
  echo "  local-to-remote   Sync generate.sh, job.sh, model/, and data/ to the remote server"
  echo "  remote-to-local   Sync synthesis/ from the remote server to local"
  echo "Example:"
  echo "  $(basename "$0") local-to-remote user@host:/path/to/synthetic_time_series_benchmarking/"
  echo "  $(basename "$0") remote-to-local user@host:/path/to/synthetic_time_series_benchmarking/"
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
fi

MODE="$1"
REMOTE="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sync_local_to_remote() {
  local items=(
    "generate.sh"
    "job.sh"
    "model"
    "data"
  )

  for item in "${items[@]}"; do
    if [[ ! -e "${SCRIPT_DIR}/${item}" ]]; then
      echo "Error: ${item} not found at ${SCRIPT_DIR}" >&2
      exit 1
    fi
  done

  echo "Syncing local to remote: ${REMOTE}"
  scp -r \
    "${SCRIPT_DIR}/generate.sh" \
    "${SCRIPT_DIR}/job.sh" \
    "${SCRIPT_DIR}/model" \
    "${SCRIPT_DIR}/data" \
    "${REMOTE}"
}

sync_remote_to_local() {
  local remote_synthesis="${REMOTE%/}"

  echo "Syncing remote to local: ${remote_synthesis} -> ${SCRIPT_DIR}/synthesis/"
  mkdir -p "${SCRIPT_DIR}/synthesis"
  scp -r "${remote_synthesis}/" "${SCRIPT_DIR}/synthesis/"
}

case "${MODE}" in
  local-to-remote)
    sync_local_to_remote
    ;;
  remote-to-local)
    sync_remote_to_local
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    ;;
esac
