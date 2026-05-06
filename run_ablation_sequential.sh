#!/usr/bin/env bash
# Run ablation permutations sequentially to avoid OOM.
# Using bash script to properly activate conda environment for vLLM multiprocessing

set -e

# GPU to use (set to 1 since GPU 0 is occupied)
export PYTHONPATH=/fs/nexus-projects/scene_graph_sd/EVA:$PYTHONPATH
export REITR_CHECKPOINT_PATH=/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models/checkpoint0149.pth
export CUDA_VISIBLE_DEVICES=1,2,3

# Activate conda environment
source /fs/nexus-scratch/mrislam/anaconda3/etc/profile.d/conda.sh
conda activate dge-t2i-env

echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8000/v1}"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/model}"
export BLIP2_MODEL_PATH="${BLIP2_MODEL_PATH:-Salesforce/blip2-itm-vit-g}"
export EVA_CLIP_CODE_DIR="/fs/nexus-projects/scene_graph_sd/EVA/EVA-CLIP/rei"
export REITR_CODE_DIR="${REITR_CODE_DIR:-/fs/nexus-projects/scene_graph_sd/RelTR}"
PROMPTS_FILE="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
echo "vLLM API: $VLLM_API_BASE"
echo "Qwen model id: $QWEN_MODEL_PATH"
echo "ReITR/RelTR code dir: $REITR_CODE_DIR"
echo "ReITR/RelTR checkpoint: ${REITR_CHECKPOINT_PATH:-unset}"
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
    # Compare all-VLM scoring with and without the second stage.
    # "V1,S2,V3"
    "V1,V2,V3"
)

RUN_NAMES=("img1" "img2")
RUN_IMAGES_DIRS=(
    "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/survey_samples/image1"
    "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/survey_samples/image2"
)
RUN_OUTPUT_DIRS=(
    "/fs/nexus-projects/scene_graph_sd/DGE-T2I/reports/ablation/permutations_img1"
    "/fs/nexus-projects/scene_graph_sd/DGE-T2I/reports/ablation/permutations_img2"
)

# Build command for each permutation
build_cmd() {
    local perm=$1
    local images_dir=$2
    local output_root=$3
    IFS=',' read -ra parts <<< "$perm"
    local p0=${parts[0]}
    local p1=${parts[1]}
    local p2=${parts[2]}

    local cmd=(
        python src/eval/ablation.py
        --output-dir "$output_root"
        --images-dir "$images_dir"
        --prompts-file "$PROMPTS_FILE"
        --low-vram
    )

    # Add vLLM when any Qwen-backed V-stage is selected.
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
        cmd+=(--e2-backend-kind blip-2 --blip2-model-path "$BLIP2_MODEL_PATH")
    elif [[ "$p1" == "S2" ]]; then
        :
    else
        cmd+=(--v2-backend-kind qwen-molmopoint --qwen-model-path "$QWEN_MODEL_PATH")
    fi

    # Stage 3
    if [[ "$p2" == "E3" ]]; then
        cmd+=("--e3-backend-kind reitr")
    else
        cmd+=(--v3-backend-kind qwen --qwen-model-path "$QWEN_MODEL_PATH")
    fi

    # Only run this specific permutation
    cmd+=(--backends "$perm")

    echo "${cmd[@]}"
}

# Check if a permutation is already completed
check_completed() {
    local perm=$1
    local output_dir=$2
    IFS=',' read -ra parts <<< "$perm"
    local filename="${output_dir}/${parts[0]}-${parts[1]}-${parts[2]}_details.json"
    if [[ -f "$filename" ]]; then
        local size=$(stat -c%s "$filename" 2>/dev/null || echo 0)
        if [[ $size -gt 10 ]]; then
            return 0  # completed
        fi
    fi
    return 1  # not completed
}

stage_permutation_outputs() {
    local output_root=$1
    local output_dir=$2

    mkdir -p "$output_dir"
    mv "$output_root"/permutations/* "$output_dir"/
}

# Main function
main() {
    # Check what's already completed
    for run_idx in "${!RUN_NAMES[@]}"; do
        local run_name=${RUN_NAMES[$run_idx]}
        local images_dir=${RUN_IMAGES_DIRS[$run_idx]}
        local output_dir=${RUN_OUTPUT_DIRS[$run_idx]}
        local completed=()

        for perm in "${PERMUTATIONS[@]}"; do
            if check_completed "$perm" "$output_dir"; then
                completed+=("$perm")
            fi
        done

        echo "Image set: $run_name"
        echo "Images dir: $images_dir"
        echo "Output dir: $output_dir"
        echo "Already completed: ${completed[*]:-none}"
        echo ""

        # Run each permutation
        for perm in "${PERMUTATIONS[@]}"; do
            # Check if already completed
            # if check_completed "$perm" "$output_dir"; then
            #     echo "Skipping $run_name $perm (already completed)"
            #     continue
            # fi

            local safe_perm=${perm//,/-}
            local output_root="./reports/ablation/.tmp_${run_name}_${safe_perm}"

            echo "=================================================="
            echo "Running image set: $run_name"
            echo "Running permutation: $perm"
            echo "=================================================="
            echo ""

            rm -rf "$output_root"

            # Build and run command
            cmd=$(build_cmd "$perm" "$images_dir" "$output_root")
            echo $cmd
            $cmd
            stage_permutation_outputs "$output_root" "$output_dir"

            echo ""
            echo "Completed: $run_name $perm"
            echo ""
        done
    done

    echo "=================================================="
    echo "All permutations complete!"
    echo "=================================================="
}

# Run main
main
