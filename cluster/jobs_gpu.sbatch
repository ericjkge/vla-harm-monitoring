#!/usr/bin/env bash
# SLURM jobs for the Kempner cluster (partition kempner, account kempner_mzitnik_lab). Usage:
#   sbatch --array=0-3 cluster/jobs.sbatch render        # 4 parallel CPU shards, ~10 min
#   sbatch cluster/jobs.sbatch rollouts_bench           # GPU, ~1 h: 15 benchmark pairs x 2 x 200 steps, videos
#   sbatch cluster/jobs.sbatch extract                  # GPU, ~10 min: activations for all 500 pairs
#   sbatch cluster/jobs.sbatch rollouts_gen             # GPU: rollouts on the first 60 generated pairs
#SBATCH --job-name=vla-harm
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=01:30:00
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
# pi0.5 runs as a local policy server in openpi's own environment; the rollout script connects to it.
# argument: checkpoint directory (default: the LIBERO-Safety checkpoint)
PI05_OFFICIAL="gs://openpi-assets/checkpoints/pi05_libero"   # Physical Intelligence's own LIBERO fine-tune; cached under OPENPI_DATA_HOME
start_pi05_server() {
  PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000)); SLOG="logs/pi05-server-${SLURM_JOB_ID:-0}.log"
  export OPENPI_DATA_HOME="$VLA_HARM_ROOT/openpi_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false
  ( cd "$VLA_HARM_ROOT/openpi" && exec .venv/bin/python scripts/serve_policy.py --port "$PORT" policy:checkpoint \
      --policy.config=pi05_libero --policy.dir="${1:-$VLA_HARM_ROOT/pi05_libero_safety}" ) > "$SLOG" 2>&1 &
  SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
  echo "pi05 server pid $SERVER_PID port $PORT, log $SLOG; waiting for it to load..."
  for i in $(seq 1 180); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "pi05 SERVER DIED. Last lines of $SLOG:"; tail -n 40 "$SLOG"; exit 1; fi
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then echo "pi05 server is up after $((i*5))s"; return 0; fi
    sleep 5
  done
  echo "pi05 server did not come up in 15 min. Last lines of $SLOG:"; tail -n 40 "$SLOG"; exit 1
}
case "${1:-}" in
  render)         $PY cluster/render_scenes.py --out "$VLA_HARM_ROOT/frames_gen" --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${SLURM_ARRAY_TASK_COUNT:-1}" ;;
  rollouts_bench) $PY cluster/rollouts.py --set benchmark --out "$VLA_HARM_ROOT/rollouts_bench" --steps 300 ;;
  sanity)         $PY cluster/rollouts.py --set sanity --limit 6 --out "$VLA_HARM_ROOT/rollouts_sanity" --steps 520 ;;
  sanity_pi05)    start_pi05_server; $PY cluster/rollouts.py --policy pi05 --port "$PORT" --set sanity --out "$VLA_HARM_ROOT/rollouts_sanity_pi05" --steps 400 ;;
  rollouts_bench_pi05) start_pi05_server; $PY cluster/rollouts.py --policy pi05 --port "$PORT" --set benchmark --out "$VLA_HARM_ROOT/rollouts_bench_pi05" --steps 300 ;;
  rollouts_skill_pi05) start_pi05_server; $PY cluster/rollouts.py --policy pi05 --port "$PORT" --set skill --out "$VLA_HARM_ROOT/rollouts_skill_pi05" --steps 300 ;;
  sanity_pi05lib) start_pi05_server "$PI05_OFFICIAL"; $PY cluster/rollouts.py --policy pi05 --controller stock --port "$PORT" --set sanity --out "$VLA_HARM_ROOT/rollouts_sanity_pi05lib" --steps 300 ;;
  rollouts_goal_pi05lib) start_pi05_server "$PI05_OFFICIAL"; $PY cluster/rollouts.py --policy pi05 --controller stock --port "$PORT" --set skill --pairs pi05lib_goal_pairs.jsonl --out "$VLA_HARM_ROOT/rollouts_goal_pi05lib" --steps 400 ;;
  rollouts_gen)   $PY cluster/rollouts.py --set generated --limit 60 --out "$VLA_HARM_ROOT/rollouts_gen" --steps 200 ;;
  extract)        $PY cluster/extract_activations.py --frames "$VLA_HARM_ROOT/frames_gen" --out "$VLA_HARM_ROOT/gen500_records.pkl" ;;
  evaluate)       $PY cluster/evaluate.py --gen "$VLA_HARM_ROOT/gen500_records.pkl" --bench "$VLA_HARM_ROOT/rollouts_bench/records.pkl" --out "$VLA_HARM_ROOT/probe_eval.png" ;;
  *) echo "usage: sbatch cluster/jobs.sbatch {render|sanity|rollouts_bench|rollouts_gen|extract|evaluate|sanity_pi05|rollouts_bench_pi05|rollouts_skill_pi05|sanity_pi05lib|rollouts_goal_pi05lib}"; exit 1 ;;
esac
