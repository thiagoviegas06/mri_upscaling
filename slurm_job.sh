#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tjv235@nyu.edu

set -euo pipefail

mkdir -p logs

export MODEL_DESTINATION="/scratch/tjv235/pytorch-example/mri-upscaling/checkpoints"
mkdir -p "${MODEL_DESTINATION}"

# Threading (good defaults)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVL="/scratch/tjv235/pytorch-example/neuro.ext3"

singularity exec --nv \
  --overlay "$OVL" \
  --bind /scratch:/scratch \
  "$SIF" /bin/bash <<'EOF'
set -euo pipefail

source /ext3/env.sh
conda activate py311

export MODEL_DESTINATION="/scratch/tjv235/pytorch-example/mri-upscaling/checkpoints"

cd /scratch/tjv235/pytorch-example/mri-upscaling/
python -u main.py
EOF