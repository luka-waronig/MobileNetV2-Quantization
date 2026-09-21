#!/bin/bash
# DAS-5 SLURM submission script for lab3_quantization.py (Quantization assignment).
#
# Assumes the "myenv" conda environment has already been created via
# create_env.sh (see envs.zip / the Lab 1 README) with requirements_das5.txt
# installed into it. Submit with:
#
#   mkdir -p logs
#   sbatch submit_lab3_quantization.sbatch

#SBATCH --job-name=lab3_quantization
#SBATCH --gres=gpu:1                 # only used to train the FP32 baseline faster (Section 1); remove for a CPU-only run
#SBATCH --constraint=GTX1080Ti       # pin to a specific GPU type: GTX1080Ti | TitanX | TitanRTX
                                      # run `sinfo -o "%N %f"` to see available node features/constraints on DAS-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00              # estimate: shorter than Lab 2 since there's no pruning fine-tune loop, just
                                      # one baseline training run + quantization (fast) + ONNX export; increase if
                                      # you raise --baseline-epochs or --train-subset-fraction and it runs long
#SBATCH --output=logs/%x_%j.out   # stdout AND stderr both land here (no --error set)

set -euo pipefail

# Data and outputs live on scratch, not $HOME (limited quota on DAS-5).
DATA_DIR="/var/scratch/$USER/lab3_data"
OUTPUT_DIR="/var/scratch/$USER/lab3_outputs"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

module purge
module load cuda12.6/toolkit/12.6

# --- Activate Virtual Environment ----------------------------------------
source /var/scratch/$USER/miniconda3/bin/activate /var/scratch/$USER/myenv

echo "===== NODE ====="
hostname

echo "===== ENVIRONMENT ====="
echo "Environment: $CONDA_PREFIX"
python --version

echo "===== GPU ====="
nvidia-smi

echo "===== PYTORCH CUDA ====="
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

# --- Run ---------------------------------------------------------------
# No --log-file here on purpose: the script logs to stdout, which SLURM
# already redirects to the single .out file above.
#
# Quantized-model measurements (Sections 2-5) run on CPU regardless of the
# GPU requested above - that's a PyTorch/ONNX Runtime int8 kernel
# limitation, not a cluster configuration choice. The GPU above only
# speeds up training the FP32 baseline in Section 1.
#
# If you already have a fine-tuned checkpoint from Lab 2, add
# "--checkpoint /var/scratch/$USER/lab2_outputs/checkpoints/<name>.pt" to
# the command below to skip training and load it directly instead.
srun python lab3_quantization.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-subset-fraction 1.0 \
    --baseline-epochs 5
