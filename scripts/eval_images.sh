#!/bin/bash
source ~/.bashrc 
conda activate dge-t2i-env
module load cuda/12.8.1
module load gcc/11.2.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model=z-image

srun python "$ROOT_DIR/src/eval/scene_graph_eval.py" --model $model
