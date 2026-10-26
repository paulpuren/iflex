#!/usr/bin/env bash
# Run one independent NS_KT trajectory and atomically pack it as HDF5.
set -euo pipefail

: "${REYNOLDS:?Set REYNOLDS}"
: "${TRAJECTORY_SEED:?Set TRAJECTORY_SEED}"
: "${NSKT_SOLVER_DIR:?Set NSKT_SOLVER_DIR to the complete solver bundle (NS_KT.py, utils.py)}"
: "${NSKT_CONFIG_TEMPLATE:?Set NSKT_CONFIG_TEMPLATE to the original 2048-resolution YAML}"

OUTPUT_DIR="${OUTPUT_DIR:-data/nskt}"
RESOLUTION="${RESOLUTION:-2048}"
# NSKT_eval reads only vorticity. With 2000 candidate indices and T=20 it
# accesses at most frames 0..138, so 140 frames retain the complete protocol.
PACK_FIELDS="${PACK_FIELDS:-w}"
PACK_MAX_FRAMES="${PACK_MAX_FRAMES:-140}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_PATH="${OUTPUT_DIR%/}/${REYNOLDS}_${RESOLUTION}_${RESOLUTION}_seed_${TRAJECTORY_SEED}.h5"

if [[ -f "${OUTPUT_PATH}" ]]; then
  echo "Validated output already exists; skipping: ${OUTPUT_PATH}"
  python -c 'import h5py,sys; f=h5py.File(sys.argv[1]); requested=int(sys.argv[2]); assert "w" in f; assert requested == 0 or f["w"].shape[0] >= requested' "${OUTPUT_PATH}" "${PACK_MAX_FRAMES}"
  exit 0
fi
[[ -f "${NSKT_SOLVER_DIR%/}/NS_KT.py" && -f "${NSKT_SOLVER_DIR%/}/utils.py" ]] || { echo "Incomplete NSKT solver bundle" >&2; exit 2; }
[[ -f "${NSKT_CONFIG_TEMPLATE}" ]] || { echo "Missing config template: ${NSKT_CONFIG_TEMPLATE}" >&2; exit 2; }

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nskt_${REYNOLDS}_${TRAJECTORY_SEED}.XXXXXX")"
trap 'rm -rf -- "${WORK_DIR}"' EXIT
cp -a "${NSKT_SOLVER_DIR%/}/." "${WORK_DIR}/"
python - "${NSKT_CONFIG_TEMPLATE}" "${WORK_DIR}/heldout.yaml" "${REYNOLDS}" "${TRAJECTORY_SEED}" <<'PY'
import sys, yaml
source, target, reynolds, seed = sys.argv[1:]
with open(source) as f:
    config = yaml.safe_load(f)
config["re"] = int(reynolds)
config["seed_number"] = int(seed)
with open(target, "w") as f:
    yaml.safe_dump(config, f, sort_keys=True)
PY
(
  cd "${WORK_DIR}"
  python NS_KT.py --config heldout.yaml
)
RESULT_DIR="${WORK_DIR}/results/DNS/${REYNOLDS}_${RESOLUTION}_${RESOLUTION}_seed_${TRAJECTORY_SEED}"
python "${REPO_ROOT}/scripts/nskt/pack_nskt_trajectory.py" \
  --result-dir "${RESULT_DIR}" --output "${OUTPUT_PATH}" --resolution "${RESOLUTION}" \
  --reynolds "${REYNOLDS}" --seed "${TRAJECTORY_SEED}" --config "${WORK_DIR}/heldout.yaml" \
  --fields ${PACK_FIELDS} --max-frames "${PACK_MAX_FRAMES}"
