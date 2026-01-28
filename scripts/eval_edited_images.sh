#!/bin/bash
source ~/.bashrc 
conda activate dge-t2i-env
module load cuda/12.8.1
module load gcc/11.2.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

models=(bagel flux_2 flux_knotex hi_dream omni qwen step1x)

for model in "${models[@]}"; do
  srun python src/eval/scene_graph_eval.py --prompts_file None --model "$model" --sg_file data/raw/prompts/edit/remove_easy_dataset.jsonl --images_dir /fs/nexus-projects/scene_graph_sd/edited_images_filtered/remove
done
