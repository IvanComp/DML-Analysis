#!/bin/bash
#SBATCH --job-name=dml-alpha05-fl-sfl
#SBATCH --account=<PROJECT_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

# Night run prepared for Leonardo.
# This script does not launch anything until you submit it with sbatch.
#
# Default study:
# - datasets: mnist cifar10 cifar100
# - models: cnn resnet vgg16
# - learning types: FL SFLV1 SFLV2
# - num clients: 100
# - clients per round: 10
# - rounds: 200
# - epochs: 1
# - repetitions: 1
# - alpha: 0.5 only
#
# Usage:
#   export PROJECT_DIR=/path/to/DML-Analysis
#   sbatch --account=<YOUR_ACCOUNT> leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch
#
# First run with venv/bootstrap:
#   sbatch --account=<YOUR_ACCOUNT> \
#     --export=ALL,PROJECT_DIR=/path/to/DML-Analysis,CREATE_VENV=1,INSTALL_REQUIREMENTS=1 \
#     leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch

PROJECT_DIR="${PROJECT_DIR:-$HOME/DML}"
RUN_MODE="${RUN_MODE:-study}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CREATE_VENV="${CREATE_VENV:-0}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
CINECA_AI_MODULE="${CINECA_AI_MODULE:-}"

STUDY_ARGS_DEFAULT=(
  --dataset mnist cifar10 cifar100
  --model cnn resnet vgg16
  --learning-type FL SFLV1 SFLV2
  --num-clients 100
  --clients-per-round 10
  --rounds 200
  --epochs 1
  --repetitions 1
  --data-distr 0.5
  --comparison-profile fair
  --device cuda
  --require-cuda
  --study-name nightly_alpha05_fl_sflv1_sflv2_100clients_10cpr_200r_1rep
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

CMD=("$PYTHON_BIN" run_experiments.py)
if [[ -n "${EXPERIMENT_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXPERIMENT_ARGS"
  CMD+=("${EXTRA_ARGS[@]}")
else
  CMD+=("${STUDY_ARGS_DEFAULT[@]}")
fi

echo "Launching command:"
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "Job finished on $(date)"
