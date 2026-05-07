#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=flux_gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-99
#SBATCH --partition=scavenger 
#SBATCH --account=scavenger 
#SBATCH --qos=scavenger
#SBATCH --nodelist=vulcan[29-45],tron[00-28,30-44,46-60],cml[18-28],gammagpu[00-17],cbcb[26-29],clip[12-13]
#SBATCH --output=slurm/flux_gen_%A.%a.out
#SBATCH --error=slurm/flux_gen_%A.%a.err

# source ~/.bashrc
# conda activate t2i-r1
# module load cuda/12.6.3

. /usr/share/Modules/init/bash
source /fs/nexus-scratch/chuonghm/miniconda3/bin/activate
conda activate compbench
module load cuda/12.4.1

MODEL_PATH="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models/FLUX1-dev"
DATA_PATH="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
EMBEDDINGS_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings/flux"
IMAGES_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/flux"
NUM_GENERATIONS=5
SEED=44

cd /fs/nexus-projects/scene_graph_sd/DGE-T2I/src/models
python flux_generate_from_embeddings.py --split_id $SLURM_ARRAY_TASK_ID --num_splits 100 --skip_existing \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --embeddings_dir $EMBEDDINGS_DIR \
    --images_dir $IMAGES_DIR \
    --num_generations $NUM_GENERATIONS \
