#!/bin/bash
source ~/.bashrc 
conda activate dge-t2i-env
module load cuda/12.8.1
module load gcc/11.2.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

model=z-image

srun python /fs/nexus-projects/scene_graph_sd/DGE-T2I/src/eval/scene_graph_eval.py --model $model