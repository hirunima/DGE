#!/bin/bash
# Script to generate images across multiple T2I models.

set -e

# Configuration
DATA_PATH="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
BASE_EMBEDDINGS_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings"
BASE_IMAGES_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images"
LOCAL_MODELS_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models"
NUM_GENERATIONS=5
SEED=44
GUIDANCE_SCALE=7.5
NEGATIVE_PROMPT=""
DEVICE_MAP="balanced"

# Model IDs (update if your repo IDs differ)
SDXL_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
SD15_MODEL_ID="stable-diffusion-v1-5/stable-diffusion-v1-5"
FLUX2_MODEL_ID="black-forest-labs/FLUX.2"
Z_IMAGE_MODEL_ID="zai-org/Z-Image"
QWEN_IMAGE_MODEL_ID="Qwen/Qwen-Image-2512"
EMU_35_MODEL_ID="BAAI/Emu3-Gen"
MOGAO_7B_MODEL_ID="Mogao/Mogao-7B"
BAGEL_MODEL_ID="BAGEL/BAGEL-7B"

# Runtime options
STEP="both"  # encode|generate|both
MODELS="" # sdxl|sd15|flux2|z-image|qwen-image-2512|emu-3-5|mogao-7b|bagel|all
START_IDX=0
END_IDX=""
NUM_SPLITS=1
SPLIT_ID=0
SKIP_EXISTING="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --end-idx)
            END_IDX="$2"
            shift 2
            ;;
        --num-splits)
            NUM_SPLITS="$2"
            shift 2
            ;;
        --split-id)
            SPLIT_ID="$2"
            shift 2
            ;;
        --num-generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device-map)
            DEVICE_MAP="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --negative-prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING="true"
            shift 1
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --step [encode|generate|both]"
            echo "  --model [sdxl|sd15|flux2|z-image|qwen-image-2512|emu-3-5|mogao-7b|bagel|all]"
            echo "  --models \"sdxl,flux2,bagel\""
            echo "  --start-idx NUM"
            echo "  --end-idx NUM"
            echo "  --num-splits NUM"
            echo "  --split-id NUM"
            echo "  --num-generations NUM"
            echo "  --seed NUM"
            echo "  --guidance-scale NUM"
            echo "  --negative-prompt TEXT"
            echo "  --device-map [balanced|auto|sequential|none]"
            echo "  --skip-existing"
            echo "  --help"
            echo ""
            echo "Examples:"
            echo "  $0 --step encode --model sdxl"
            echo "  $0 --step generate --model sdxl"
            echo "  $0 --step both --model all --start-idx 0 --end-idx 100"
            echo "  $0 --step generate --model sdxl --num-splits 100 --split-id 0"
            echo "  $0 --step generate --models \"sdxl,flux2,bagel\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

has_model() {
    local key="$1"
    if [ "$MODELS" = "all" ]; then
        return 0
    fi
    local needle=",$key,"
    local haystack=",$MODELS,"
    if [ "${haystack#*$needle}" != "$haystack" ]; then
        return 0
    fi
    # fi
    return 1
}

resolve_model_path() {
    local model_id="$1"
    local local_dir="$2"
    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir")" ]; then
        echo "$local_dir"
    else
        echo "$model_id"
    fi
}

images_complete() {
    local images_dir="$1"
    python - <<PY
import json
import os
import sys

data_path = "$DATA_PATH"
images_dir = "$images_dir"
num_generations = int("$NUM_GENERATIONS")
start_idx = "$START_IDX"
end_idx = "$END_IDX"
num_splits = int("$NUM_SPLITS")
split_id = int("$SPLIT_ID")

if not os.path.isdir(images_dir):
    sys.exit(1)

with open(data_path, "r") as f:
    data = json.load(f)

def resolve_indices(data_len, start_idx, end_idx, num_splits, split_id):
    if start_idx != "" and end_idx != "":
        return int(start_idx), int(end_idx)
    items_per_split = data_len // num_splits
    remainder = data_len % num_splits
    if split_id < remainder:
        start = split_id * (items_per_split + 1)
        end = start + items_per_split + 1
    else:
        start = split_id * items_per_split + remainder
        end = start + items_per_split
    return start, end

start, end = resolve_indices(len(data), start_idx, end_idx, num_splits, split_id)
for idx in range(start, end):
    example_id = f"{idx:04d}"
    for i in range(1, num_generations + 1):
        if not os.path.exists(os.path.join(images_dir, f"{example_id}-{i}.png")):
            sys.exit(1)
sys.exit(0)
PY
}

embeddings_complete() {
    local embeddings_dir="$1"
    python - <<PY
import json
import os
import sys

data_path = "$DATA_PATH"
embeddings_dir = "$embeddings_dir"
start_idx = "$START_IDX"
end_idx = "$END_IDX"
num_splits = int("$NUM_SPLITS")
split_id = int("$SPLIT_ID")

if not os.path.isdir(embeddings_dir):
    sys.exit(1)

with open(data_path, "r") as f:
    data = json.load(f)

def resolve_indices(data_len, start_idx, end_idx, num_splits, split_id):
    if start_idx != "" and end_idx != "":
        return int(start_idx), int(end_idx)
    items_per_split = data_len // num_splits
    remainder = data_len % num_splits
    if split_id < remainder:
        start = split_id * (items_per_split + 1)
        end = start + items_per_split + 1
    else:
        start = split_id * items_per_split + remainder
        end = start + items_per_split
    return start, end

start, end = resolve_indices(len(data), start_idx, end_idx, num_splits, split_id)
for idx in range(start, end):
    example_id = f"{idx:04d}"
    if not os.path.exists(os.path.join(embeddings_dir, f"{example_id}.pt")):
        sys.exit(1)
sys.exit(0)
PY
}

encode_model() {
    local model_key="$1"
    local model_id="$2"
    local embeddings_dir="$3"

    echo "========================================="
    echo "Encoding: $model_key"
    echo "Model ID: $model_id"
    echo "Output: $embeddings_dir"
    echo "========================================="

    CMD="python /fs/nexus-projects/scene_graph_sd/DGE-T2I/src/models/encode_prompts.py \
        --model_id $model_id \
        --data_path $DATA_PATH \
        --embeddings_dir $embeddings_dir \
        --guidance_scale $GUIDANCE_SCALE \
        --negative_prompt \"$NEGATIVE_PROMPT\" \
        --device-map $DEVICE_MAP \
        --num_splits $NUM_SPLITS \
        --split_id $SPLIT_ID \
        --start_idx $START_IDX"

    if [ -n "$END_IDX" ]; then
        CMD="$CMD --end_idx $END_IDX"
    fi
    if [ "$SKIP_EXISTING" = "true" ]; then
        CMD="$CMD --skip_existing"
    fi

    eval $CMD
    echo ""
}

generate_model() {
    local model_key="$1"
    local model_id="$2"
    local embeddings_dir="$3"
    local images_dir="$4"

    echo "========================================="
    echo "Generating: $model_key"
    echo "Model ID: $model_id"
    echo "Output: $images_dir"
    echo "========================================="

    CMD="python /fs/nexus-projects/scene_graph_sd/DGE-T2I/src/models/generate_images_from_embeddings.py \
        --model_id $model_id \
        --data_path $DATA_PATH \
        --embeddings_dir $embeddings_dir \
        --images_dir $images_dir \
        --num_generations $NUM_GENERATIONS \
        --seed $SEED \
        --guidance_scale $GUIDANCE_SCALE \
        --device-map $DEVICE_MAP \
        --num_splits $NUM_SPLITS \
        --split_id $SPLIT_ID \
        --start_idx $START_IDX"

    if [ -n "$END_IDX" ]; then
        CMD="$CMD --end_idx $END_IDX"
    fi
    if [ "$SKIP_EXISTING" = "true" ]; then
        CMD="$CMD --skip_existing"
    fi

    eval $CMD
    echo ""
}

if has_model "sdxl"; then
    echo "got sdxl"
fi

if has_model "sd15"; then
    echo "got sd15" 
fi

if has_model "flux2"; then
    echo "got flux2"
fi

if has_model "z-image"; then
    echo "got z-image"
fi

if has_model "qwen-image-2512"; then
    echo "got qwen-image-2512"
fi

if has_model "emu-3-5"; then
    echo "got emu-3-5"
fi

if has_model "mogao-7b"; then
    echo "got mogao-7b"
fi

if has_model "bagel"; then
    echo "got bagel"
fi

echo "========================================="
echo "Generation complete!"
echo "Base output dir: $BASE_IMAGES_DIR"
echo "========================================="

