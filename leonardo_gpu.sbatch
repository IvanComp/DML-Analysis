#!/bin/bash
#SBATCH --job-name=dml-leonardo
#SBATCH --account=<PROJECT_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

# Usage:
#   1. Edit PROJECT_DIR and --account if needed.
#   2. Optionally export RUN_MODE=single to run flower_baseline.py instead of run_experiments.py.
#   3. Submit with: sbatch leonardo_gpu.sbatch

PROJECT_DIR="${PROJECT_DIR:-$HOME/DML}"
RUN_MODE="${RUN_MODE:-study}"               # study | single
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CREATE_VENV="${CREATE_VENV:-0}"             # 1 = create the venv if missing
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"  # 1 = pip install -r requirements.txt

# If not set, the script loads the newest cineca-ai module available in profile/deeplrn.
CINECA_AI_MODULE="${CINECA_AI_MODULE:-}"

# Default arguments for repeated studies.
STUDY_ARGS_DEFAULT=(
  --dataset mnist cifar10 stl10
  --model cnn resnet vgg16
  --learning-type FL
  --num-clients 100
  --clients-per-round 10
  --rounds 200
  --epochs 1
  --repetitions 1
  --data-distr 0.01 0.5 1.0
  --study-name fl_100clients_10perround_200rounds_1rep
)

# Default arguments for a single run.
SINGLE_ARGS_DEFAULT=(
  --dataset mnist
  --model cnn
  --baseline fedavg
  --learning-type FL
  --num_clients 20
  --clients-per-round 10
  --rounds 10
  --epochs 1
  --device cuda
)

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm"
cd "$PROJECT_DIR"

module purge
module load profile/deeplrn

if [[ -z "$CINECA_AI_MODULE" ]]; then
  CINECA_AI_MODULE="$(module -t av cineca-ai 2>&1 | awk '/^cineca-ai\// {print $1}' | sort -V | tail -n 1)"
fi

if [[ -z "$CINECA_AI_MODULE" ]]; then
  echo "No cineca-ai module found after loading profile/deeplrn." >&2
  exit 1
fi

module load "$CINECA_AI_MODULE"

if [[ -d "$VENV_DIR" ]]; then
  source "$VENV_DIR/bin/activate"
elif [[ "$CREATE_VENV" == "1" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages
  source "$VENV_DIR/bin/activate"
else
  echo "Virtual environment not found: $VENV_DIR" >&2
  echo "Create it first, or resubmit with CREATE_VENV=1." >&2
  exit 1
fi

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Job started on $(date)"
echo "Node list: ${SLURM_JOB_NODELIST:-unknown}"
echo "Working directory: $PROJECT_DIR"
echo "Python: $("$PYTHON_BIN" --version 2>&1)"
echo "Loaded cineca-ai module: $CINECA_AI_MODULE"

nvidia-smi

if [[ "$RUN_MODE" == "study" ]]; then
  CMD=("$PYTHON_BIN" run_experiments.py)
  if [[ -n "${EXPERIMENT_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "$EXPERIMENT_ARGS"
    CMD+=("${EXTRA_ARGS[@]}")
  else
    CMD+=("${STUDY_ARGS_DEFAULT[@]}")
  fi
elif [[ "$RUN_MODE" == "single" ]]; then
  CMD=("$PYTHON_BIN" flower_baseline.py)
  if [[ -n "${EXPERIMENT_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "$EXPERIMENT_ARGS"
    CMD+=("${EXTRA_ARGS[@]}")
  else
    CMD+=("${SINGLE_ARGS_DEFAULT[@]}")
  fi
else
  echo "Unsupported RUN_MODE: $RUN_MODE (use study or single)." >&2
  exit 1
fi

echo "Launching command:"
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "Job finished on $(date)"
