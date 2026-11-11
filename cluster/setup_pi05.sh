#!/usr/bin/env bash
# One-time setup for the LIBERO-Safety pi0.5 checkpoint (run on a login node AFTER cluster/setup.sh).
#   bash cluster/setup_pi05.sh
# pi0.5 runs in its own environment (openpi: JAX, its own pins) and is served over a local websocket; the simulator
# stays in the existing environment and talks to it with openpi's small client package. This is openpi's own
# recommended split, and it avoids mixing two incompatible dependency sets.
#   $VLA_HARM_ROOT/openpi/                openpi checkout + its .venv (~10 GB)
#   $VLA_HARM_ROOT/pi05_libero_safety/    the checkpoint (~12.5 GB)
set -euo pipefail
ROOT="${VLA_HARM_ROOT:-$HOME/vla-harm}"; PY="$ROOT/env/bin/python"
[ -x "$PY" ] || { echo "run cluster/setup.sh first ($PY not found)"; exit 1; }
export PIP_USER=0 UV_CACHE_DIR="$ROOT/uv_cache" UV_PYTHON_INSTALL_DIR="$ROOT/uv_python" HF_HOME="$ROOT/hf_cache" GIT_LFS_SKIP_SMUDGE=1
cd "$ROOT"

# 1. openpi + its environment (uv is installed into the existing env only as a tool)
$PY -m pip install -q uv
[ -d openpi ] || git clone -q https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout -q 215abfb   # the commit these scripts were written against
# rerun-sdk (a visualizer pulled in by lerobot, unused for inference) only ships wheels for glibc >= 2.31; the
# cluster has 2.28, so skip it. `uv sync` already installs openpi itself, so no separate `pip install -e .`.
"$ROOT/env/bin/uv" sync --python "$PY" --no-install-package rerun-sdk
cd "$ROOT"

# 2. checkpoint. Its normalization stats are stored under assets/lerobot, but openpi's pi05_libero config looks for
#    assets/physical-intelligence/libero, so link one to the other.
$PY - <<PY
from huggingface_hub import snapshot_download
snapshot_download("LIBERO-Safety/pi05_libero_safety", local_dir="$ROOT/pi05_libero_safety")
PY
mkdir -p pi05_libero_safety/assets/physical-intelligence
ln -sfn ../lerobot pi05_libero_safety/assets/physical-intelligence/libero
test -f pi05_libero_safety/assets/physical-intelligence/libero/norm_stats.json

# 2b. Physical Intelligence's official pi0.5 LIBERO checkpoint (~12 GB), fetched into openpi's cache so jobs do not
#     have to download it. It was trained on original LIBERO, where one scene has many tasks told apart by wording.
OPENPI_DATA_HOME="$ROOT/openpi_cache" "$ROOT/openpi/.venv/bin/python" -c "from openpi.shared import download; print('official checkpoint at', download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"

# 3. client package in the simulator environment (--no-deps: it pins numpy<2 only out of caution)
$PY -m pip install -q msgpack websockets typing_extensions
$PY -m pip install -q --no-deps -e openpi/packages/openpi-client

# 4. checks
"$ROOT/openpi/.venv/bin/python" -c "import jax, openpi; from openpi.training import config; from openpi.policies import policy_config; from openpi.serving import websocket_policy_server; config.get_config('pi05_libero'); print('openpi OK: jax', jax.__version__)"
$PY -c "from openpi_client import websocket_client_policy, image_tools; print('client OK')"
du -sh pi05_libero_safety openpi/.venv
echo "pi05 setup done."
