#!/bin/bash
source ~/.bashrc 
conda activate dge-t2i-env
module load cuda/12.8.1
module load gcc/11.2.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

samples=2

srun python -m src.data.generate --samples $samples