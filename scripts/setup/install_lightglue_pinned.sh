#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="eb42fee2d71449efb0aa5c10549752b5d75384d8"
OFFICIAL_REPO="https://github.com/cvg/LightGlue.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${REPO_ROOT}/third_party/LightGlue"
PYTHON_BIN="${PYTHON:-python}"

echo "Repository root : ${REPO_ROOT}"
echo "Python          : $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Python version  : $(${PYTHON_BIN} -c 'import sys; print(sys.version.split()[0])')"
echo "Target checkout : ${TARGET}"
echo "Pinned commit   : ${PINNED_COMMIT}"

"${PYTHON_BIN}" - <<'PY'
import sys
print("Python prefix   :", sys.prefix)
if ".drone_venv" not in sys.prefix:
    print(
        "WARNING: current Python does not appear to be .drone_venv. "
        "Activate source .drone_venv/bin/activate first if this is unintended."
    )
PY

mkdir -p "${REPO_ROOT}/third_party"

if [[ ! -e "${TARGET}" ]]; then
    echo
    echo "[1/5] Cloning official LightGlue repository..."
    git clone "${OFFICIAL_REPO}" "${TARGET}"
else
    echo
    echo "[1/5] Existing target found; validating..."
    if [[ ! -d "${TARGET}/.git" ]]; then
        echo "ERROR: ${TARGET} exists but is not a Git checkout." >&2
        echo "Move/remove that directory and rerun this installer." >&2
        exit 2
    fi

    ORIGIN="$(git -C "${TARGET}" remote get-url origin 2>/dev/null || true)"
    case "${ORIGIN}" in
        *github.com/cvg/LightGlue.git|*github.com/cvg/LightGlue)
            ;;
        *)
            echo "ERROR: existing LightGlue checkout has unexpected origin:" >&2
            echo "  ${ORIGIN}" >&2
            echo "Expected official cvg/LightGlue repository." >&2
            exit 3
            ;;
    esac

    DIRTY="$(git -C "${TARGET}" status --porcelain --untracked-files=no)"
    if [[ -n "${DIRTY}" ]]; then
        echo "ERROR: tracked files in ${TARGET} are modified." >&2
        echo "Commit/stash/revert those changes before installing the pinned research dependency." >&2
        exit 4
    fi
fi

echo
echo "[2/5] Fetching and checking out pinned commit..."
git -C "${TARGET}" fetch origin "${PINNED_COMMIT}"
git -C "${TARGET}" checkout --detach "${PINNED_COMMIT}"

HEAD="$(git -C "${TARGET}" rev-parse HEAD)"
if [[ "${HEAD}" != "${PINNED_COMMIT}" ]]; then
    echo "ERROR: LightGlue HEAD ${HEAD} != pinned ${PINNED_COMMIT}" >&2
    exit 5
fi

echo
echo "[3/5] Installing pinned checkout editable into active Python..."
"${PYTHON_BIN}" -m pip install -e "${TARGET}"

echo
echo "[4/5] Verifying Python resolves LightGlue from the vendored checkout..."
LIGHTGLUE_TARGET="${TARGET}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import lightglue
target = Path(os.environ["LIGHTGLUE_TARGET"]).resolve()
module_path = Path(lightglue.__file__).resolve()
package_root = module_path.parent.parent
print("lightglue.__file__:", module_path)
print("resolved package root:", package_root)
try:
    inside = module_path.is_relative_to(target)
except AttributeError:
    inside = str(module_path).startswith(str(target) + os.sep)
if not inside:
    raise SystemExit(
        "ERROR: Python imported LightGlue from a different location. "
        f"Expected under {target}, got {module_path}"
    )
PY

echo
echo "[5/5] Prefetching weights and validating project adapter..."
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
LIGHTGLUE_TARGET="${TARGET}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
from uavloc.geometry.lightglue_absolute_frontend import SuperPointLightGlueAbsoluteFrontend
frontend = SuperPointLightGlueAbsoluteFrontend(
    Path(os.environ["LIGHTGLUE_TARGET"]),
    device_name="cpu",
)
meta = frontend.describe()
print(json.dumps(meta, indent=2, sort_keys=True))
if meta["source_mode"] != "git_checkout":
    raise SystemExit(
        f"ERROR: expected source_mode=git_checkout, got {meta['source_mode']}"
    )
if meta["resolved_git_commit"] != meta["pinned_commit"]:
    raise SystemExit("ERROR: resolved LightGlue commit does not match pinned commit")
PY

echo
echo "PASS_LIGHTGLUE_PINNED_INSTALL"
echo "Next:"
echo "  python scripts/villoc/geometry/lf2_0a_lightglue_absolute_preflight_smoke.py \\"
echo "    --config configs/research/learned_geometric_frontend_lf2_0a.yaml"
