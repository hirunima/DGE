#!/usr/bin/env bash
# Run ablation permutations sequentially to avoid OOM.
# Using bash script to properly activate conda environment for vLLM multiprocessing

set -e

# GPU to use (set to 1 since GPU 0 is occupied)
export CUDA_VISIBLE_DEVICES=1

# Activate conda environment
source /fs/nexus-scratch/mrislam/anaconda3/etc/profile.d/conda.sh
conda activate dge-t2i-env

echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8000/v1}"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/model}"
echo "vLLM API: $VLLM_API_BASE"
echo "Qwen model id: $QWEN_MODEL_PATH"
echo ""

# V1 permutations first (4), then E1 permutations (4)
# V1: ~1.5 hours, E1: ~15 min = ~2 hours total for 106 images
# PERMUTATIONS=(
#     # V1 permutations (slower - uses Qwen for node detection)
#     "V1,E2,E3"
#     "V1,E2,V3"
#     "V1,V2,E3"
#     "V1,V2,V3"
#     # E1 permutations (faster - uses GroundingDINO)
#     "E1,E2,E3"
#     "E1,E2,V3"
#     "E1,V2,E3"
#     "E1,V2,V3"
# )

PERMUTATIONS=(
    # # V1 permutations (slower - uses Qwen for node detection)
    "V1,E2,E3,V3"
    "V1,V2,E3,V3"
    # # E1 permutations (faster - uses GroundingDINO)
    "E1,E2,E3,V3"
    "E1,V2,E3,V3"
)


# Build command for each permutation
build_cmd() {
    local perm=$1
    IFS=',' read -ra parts <<< "$perm"
    local p0=${parts[0]}
    local p1=${parts[1]}
    local p2=${parts[2]}

    # TODO: run on image2!!
    local cmd=(
        python src/eval/ablation.py
        --output-dir ./reports/ablation
        --images-dir /fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/survey_samples/image2
        --prompts-file /fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json
        --low-vram
    )

    Add vLLM when any Qwen-backed V-stage is selected.
    if [[ "$p0" == V1 || "$p1" == V2 || "$p2" == V3 || "${parts[3]:-}" == V3 ]]; then
        cmd+=(--use-vllm)
    fi

    # Stage 1 - only set the backend kind for the one being used
    if [[ "$p0" == "E1" ]]; then
        cmd+=(--e1-backend-kind grounding-dino --eupe-model-path IDEA-Research/grounding-dino-base)
    else
        cmd+=(--v1-backend-kind qwen --qwen-model-path "$QWEN_MODEL_PATH")
    fi

    # Stage 2
    if [[ "$p1" == "E2" ]]; then
        cmd+=(--e2-backend-kind siglip --siglip-model-path google/siglip2-so400m-patch14-384)
    else
        cmd+=(--v2-backend-kind qwen --qwen-model-path "$QWEN_MODEL_PATH")
    fi

    # Stage 3
    # if [[ "$p2" == "E3" ]]; then
        cmd+=(--e3-backend-kind siglip --siglip-model-path google/siglip2-so400m-patch14-384)
    # else
        cmd+=(--v3-backend-kind qwen --qwen-model-path "$QWEN_MODEL_PATH")
    # fi

    # Only run this specific permutation
    cmd+=(--backends "$perm")

    echo "${cmd[@]}"
}

# Check if a permutation is already completed
check_completed() {
    local perm=$1
    IFS=',' read -ra parts <<< "$perm"
    local filename="./reports/ablation/permutations/${parts[0]}-${parts[1]}-${parts[2]}_details.json"
    if [[ -f "$filename" ]]; then
        local size=$(stat -c%s "$filename" 2>/dev/null || echo 0)
        if [[ $size -gt 10 ]]; then
            return 0  # completed
        fi
    fi
    return 1  # not completed
}

# Main function
main() {
    # Check what's already completed
    local completed=()
    for perm in "${PERMUTATIONS[@]}"; do
        if check_completed "$perm"; then
            completed+=("$perm")
        fi
    done

    echo "Already completed: ${completed[*]:-none}"
    echo ""

    # Run each permutation
    for perm in "${PERMUTATIONS[@]}"; do
        # Check if already completed
        if check_completed "$perm"; then
            echo "Skipping $perm (already completed)"
            continue
        fi

        echo "=================================================="
        echo "Running permutation: $perm"
        echo "=================================================="
        echo ""

        # Build and run command
        cmd=$(build_cmd "$perm")
        echo $cmd
        $cmd

        echo ""
        echo "Completed: $perm"
        echo ""
    done

    echo "=================================================="
    echo "All permutations complete!"
    echo "=================================================="
}

# Run main
main
