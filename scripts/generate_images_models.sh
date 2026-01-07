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
MODEL="all"  # sdxl|sd15|flux2|z-image|qwen-image-2512|emu-3-5|mogao-7b|bagel|all
MODELS=""
START_IDX=0
END_IDX=""
NUM_SPLITS=1
SPLIT_ID=0
SKIP_EXISTING="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
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

if [ -n "$MODELS" ]; then
    MODEL="all"
fi

has_model() {
    local key="$1"
    if [ "$MODEL" = "all" ]; then
        return 0
    fi
    if [ "$MODEL" = "$key" ]; then
        return 0
    fi
    if [ -n "$MODELS" ]; then
        local needle=",$key,"
        local haystack=",$MODELS,"
        if [ "${haystack#*$needle}" != "$haystack" ]; then
            return 0
        fi
    fi
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
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/sdxl"; then
        echo "SDXL embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/sdxl"; then
        echo "SDXL images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/sdxl" && images_complete "$BASE_IMAGES_DIR/sdxl"; then
        echo "SDXL embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$SDXL_MODEL_ID" "$LOCAL_MODELS_DIR/sdxl")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "SDXL" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/sdxl"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "SDXL" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/sdxl" "$BASE_IMAGES_DIR/sdxl"
        fi
    fi
fi

if has_model "sd15"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/sd15"; then
        echo "SD-1.5 embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/sd15"; then
        echo "SD-1.5 images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/sd15" && images_complete "$BASE_IMAGES_DIR/sd15"; then
        echo "SD-1.5 embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$SD15_MODEL_ID" "$LOCAL_MODELS_DIR/sd15")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "SD-1.5" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/sd15"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "SD-1.5" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/sd15" "$BASE_IMAGES_DIR/sd15"
        fi
    fi
fi

if has_model "flux2"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/flux2"; then
        echo "FLUX2.0 embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/flux2"; then
        echo "FLUX2.0 images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/flux2" && images_complete "$BASE_IMAGES_DIR/flux2"; then
        echo "FLUX2.0 embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$FLUX2_MODEL_ID" "$LOCAL_MODELS_DIR/flux2")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "FLUX2.0" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/flux2"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "FLUX2.0" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/flux2" "$BASE_IMAGES_DIR/flux2"
        fi
    fi
fi

if has_model "z-image"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/z-image"; then
        echo "Z-Image embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/z-image"; then
        echo "Z-Image images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/z-image" && images_complete "$BASE_IMAGES_DIR/z-image"; then
        echo "Z-Image embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$Z_IMAGE_MODEL_ID" "$LOCAL_MODELS_DIR/z-image")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "Z-Image" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/z-image"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "Z-Image" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/z-image" "$BASE_IMAGES_DIR/z-image"
        fi
    fi
fi

if has_model "qwen-image-2512"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/qwen-image-2512"; then
        echo "Qwen-Image-2512 embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/qwen-image-2512"; then
        echo "Qwen-Image-2512 images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/qwen-image-2512" && images_complete "$BASE_IMAGES_DIR/qwen-image-2512"; then
        echo "Qwen-Image-2512 embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$QWEN_IMAGE_MODEL_ID" "$LOCAL_MODELS_DIR/qwen-image-2512")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "Qwen-Image-2512" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/qwen-image-2512"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "Qwen-Image-2512" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/qwen-image-2512" "$BASE_IMAGES_DIR/qwen-image-2512"
        fi
    fi
fi

if has_model "emu-3-5"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/emu-3-5"; then
        echo "Emu 3.5 embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/emu-3-5"; then
        echo "Emu 3.5 images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/emu-3-5" && images_complete "$BASE_IMAGES_DIR/emu-3-5"; then
        echo "Emu 3.5 embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$EMU_35_MODEL_ID" "$LOCAL_MODELS_DIR/emu-3-5")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "Emu 3.5" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/emu-3-5"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "Emu 3.5" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/emu-3-5" "$BASE_IMAGES_DIR/emu-3-5"
        fi
    fi
fi

if has_model "mogao-7b"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/mogao-7b"; then
        echo "Mogao-7B embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/mogao-7b"; then
        echo "Mogao-7B images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/mogao-7b" && images_complete "$BASE_IMAGES_DIR/mogao-7b"; then
        echo "Mogao-7B embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$MOGAO_7B_MODEL_ID" "$LOCAL_MODELS_DIR/mogao-7b")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "Mogao-7B" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/mogao-7b"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "Mogao-7B" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/mogao-7b" "$BASE_IMAGES_DIR/mogao-7b"
        fi
    fi
fi

if has_model "bagel"; then
    SKIP_MODEL="false"
    if [ "$STEP" = "encode" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/bagel"; then
        echo "BAGEL embeddings already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "generate" ] && images_complete "$BASE_IMAGES_DIR/bagel"; then
        echo "BAGEL images already complete; skipping download and steps."
        SKIP_MODEL="true"
    elif [ "$STEP" = "both" ] && embeddings_complete "$BASE_EMBEDDINGS_DIR/bagel" && images_complete "$BASE_IMAGES_DIR/bagel"; then
        echo "BAGEL embeddings and images already complete; skipping download and steps."
        SKIP_MODEL="true"
    fi
    if [ "$SKIP_MODEL" = "false" ]; then
        MODEL_PATH=$(resolve_model_path "$BAGEL_MODEL_ID" "$LOCAL_MODELS_DIR/bagel")
        if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
            encode_model "BAGEL" "$MODEL_PATH" "$BASE_EMBEDDINGS_DIR/bagel"
        fi
        if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
            generate_model "BAGEL" "$MODEL_PATH" \
                "$BASE_EMBEDDINGS_DIR/bagel" "$BASE_IMAGES_DIR/bagel"
        fi
    fi
fi

echo "========================================="
echo "Generation complete!"
echo "Base output dir: $BASE_IMAGES_DIR"
echo "========================================="
