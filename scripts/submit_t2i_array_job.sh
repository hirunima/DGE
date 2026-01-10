#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=t2i_models
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --array=0-99
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodelist=vulcan[29-45],tron[00-28,30-44,46-60],cml[18-28],gammagpu[00-17],cbcb[26-29],clip[12-13]
#SBATCH --output=logs/%A/t2i_models_%a.out
#SBATCH --error=logs/%A/t2i_models_%a.err

. /usr/share/Modules/init/bash
source ~/.bashrc
conda activate dge-t2i-env
module load cuda/12.6.3

ROOT_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I"
MODEL="qwen-image-2512"  # sdxl|sd15|flux2|z-image|qwen-image-2512|emu-3-5|mogao-7b|bagel|all
STEP="both"  # encode|generate|both
NUM_SPLITS=100

cd "$ROOT_DIR"

scripts/generate_images_models.sh \
    --step "$STEP" \
    --models "$MODEL" \
    --num-splits "$NUM_SPLITS" \
    --split-id "$SLURM_ARRAY_TASK_ID" \
    --device-map "balanced" \
    --encode-device-map "none" \
    --skip-existing
