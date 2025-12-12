#!/bin/bash
source ~/.bashrc 
conda activate t2i-r1
module load cuda/12.6.3
module load gcc/11.2.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
srun python -m src.data.generate --input_dir test_input 