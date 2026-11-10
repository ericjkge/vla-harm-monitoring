#!/usr/bin/env bash
# SLURM jobs for the Kempner cluster (partition kempner, account kempner_mzitnik_lab). Usage:
#   sbatch --array=0-3 cluster/jobs.sbatch render        # 4 parallel CPU shards, ~10 min
#   sbatch cluster/jobs.sbatch rollouts_bench           # GPU, ~1 h: 15 benchmark pairs x 2 x 200 steps, videos
#   sbatch cluster/jobs.sbatch extract                  # GPU, ~10 min: activations for all 500 pairs
#   sbatch cluster/jobs.sbatch rollouts_gen             # GPU: rollouts on the first 60 generated pairs
#SBATCH --job-name=vla-harm
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j-%a.out
set -euo pipefail
mkdir -p logs
module load ncf/1.0.0-fasrc01 2>/dev/null || true
module load miniconda3/py39_4.11.0-linux_x64-ncf 2>/dev/null || true
export VLA_HARM_ROOT="${VLA_HARM_ROOT:-$HOME/vla-harm}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HOME="${VLA_HARM_ROOT}/hf_cache"
PY="$VLA_HARM_ROOT/env/bin/python"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # under sbatch, $0 is a spool copy, so use the submit dir (run sbatch from the repo root)
echo "node $(hostname)  job ${SLURM_JOB_ID:-}  gpus ${CUDA_VISIBLE_DEVICES:-none}"; nvidia-smi --query-gpu=name,memory.used,memory.total,compute_mode --format=csv || true
case "${1:-}" in
  render)         $PY cluster/render_scenes.py --out "$VLA_HARM_ROOT/frames_gen" --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${SLURM_ARRAY_TASK_COUNT:-1}" ;;
  rollouts_bench) $PY cluster/rollouts.py --set benchmark --out "$VLA_HARM_ROOT/rollouts_bench" --steps 200 ;;
  sanity)         $PY cluster/rollouts.py --set sanity --limit 6 --out "$VLA_HARM_ROOT/rollouts_sanity" --steps 480 ;;
  rollouts_gen)   $PY cluster/rollouts.py --set generated --limit 60 --out "$VLA_HARM_ROOT/rollouts_gen" --steps 200 ;;
  extract)        $PY cluster/extract_activations.py --frames "$VLA_HARM_ROOT/frames_gen" --out "$VLA_HARM_ROOT/gen500_records.pkl" ;;
  evaluate)       $PY cluster/evaluate.py --gen "$VLA_HARM_ROOT/gen500_records.pkl" --bench "$VLA_HARM_ROOT/rollouts_bench/records.pkl" --out "$VLA_HARM_ROOT/probe_eval.png" ;;
  *) echo "usage: sbatch cluster/jobs.sbatch {render|sanity|rollouts_bench|rollouts_gen|extract|evaluate}"; exit 1 ;;
esac
