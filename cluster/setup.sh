#!/usr/bin/env bash
# One-time environment setup on the cluster. Run from a login node (or a CPU job):
#   bash cluster/setup.sh
# Creates a conda env "vla-harm" with LIBERO-Safety, robosuite, mujoco, and the OpenVLA stack,
# downloads the simulator assets (~10 GB) and writes ~/.libero/config.yaml.
set -euo pipefail
# Everything (conda env, HF model cache, simulator assets, outputs) lives under ROOT so a full home quota
# is never a problem. On FASRC point this at scratch, e.g.:
#   export VLA_HARM_ROOT=/n/netscratch/mzitnik_lab/Lab/$USER/vla-harm
ROOT="${VLA_HARM_ROOT:-$HOME/vla-harm}"
mkdir -p "$ROOT"; cd "$ROOT"
export HF_HOME="$ROOT/hf_cache" CONDA_PKGS_DIRS="$ROOT/conda_pkgs" PIP_CACHE_DIR="$ROOT/pip_cache"

# FASRC/Kempner conda modules (from the lab's cluster guide); harmless elsewhere
module load ncf/1.0.0-fasrc01 2>/dev/null || true
module load miniconda3/py39_4.11.0-linux_x64-ncf 2>/dev/null || true
[ -x "$ROOT/env/bin/python" ] || conda create -y -p "$ROOT/env" python=3.11
# Use the env's own interpreter/pip by full path: `conda activate` is unreliable inside scripts and
# silently falls back to the read-only system pip, which then installs into ~/.local.
PY="$ROOT/env/bin/python"; PIP="$PY -m pip"
export PIP_USER=0

[ -d LIBERO-Safety ] || git clone -q https://github.com/LIBERO-SAFETY/LIBERO-Safety.git
cd LIBERO-Safety
# OpenVLA-compatible stack first; the repo's requirements.txt pins wandb 0.13 (dead 'pathtools') and
# transformers 4.21 (breaks OpenVLA), so it is not used.
$PIP install -q "torch>=2.2" "huggingface_hub<1.0" tokenizers==0.19.1 transformers==4.40.1 timm==0.9.10 accelerate scikit-learn
$PIP install -q bddl==1.0.1 easydict gym cloudpickle termcolor wand scikit-image h5py "imageio[ffmpeg]" "mujoco>=3.2,<3.3" matplotlib pandas
$PIP install -q third_party/robosuite-1.4
$PIP install -q -e .

$PY - <<'PY'
from huggingface_hub import hf_hub_download
import zipfile, yaml, pathlib, os
z = hf_hub_download("LIBERO-Safety/libero_safety_assets", "assets.zip", repo_type="dataset")
root = pathlib.Path("libero/libero")
if not (root/"assets").exists():
    with zipfile.ZipFile(z) as f: f.extractall(root)
cfg = pathlib.Path.home()/".libero"; cfg.mkdir(exist_ok=True)
yaml.safe_dump({"assets": str(root.resolve()/"assets"), "bddl_files": str(root.resolve()/"bddl_files"),
                "benchmark_root": str(root.resolve()), "datasets": str(root.resolve().parent/"datasets"),
                "init_states": str(root.resolve()/"init_files")}, open(cfg/"config.yaml", "w"))
print("assets + config ready")
PY
$PY -c "import robosuite, mujoco, libero, transformers; print('env OK: mujoco', mujoco.__version__, 'transformers', transformers.__version__)"
echo "setup done. Export VLA_HARM_ROOT=$ROOT before submitting jobs (add it to ~/.bashrc)."
