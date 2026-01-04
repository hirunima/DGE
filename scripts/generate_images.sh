#!/bin/bash
# Script to pre-encode prompts and generate images with FLUX

set -e  # Exit on error

# Configuration
MODEL_PATH="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/models/FLUX1-dev"
DATA_PATH="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
EMBEDDINGS_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/embeddings/flux"
IMAGES_DIR="/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/flux"
NUM_GENERATIONS=5
SEED=44

# Parse command line arguments
STEP="both"  # Options: encode, generate, both
START_IDX=0
END_IDX=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --step [encode|generate|both]  Which step to run (default: both)"
            echo "  --start-idx NUM                 Start index for processing (default: 0)"
            echo "  --end-idx NUM                   End index for processing (default: all)"
            echo "  --help                          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run both steps for all prompts"
            echo "  $0 --step encode                # Only encode prompts"
            echo "  $0 --step generate              # Only generate images (assumes embeddings exist)"
            echo "  $0 --start-idx 0 --end-idx 100  # Process only examples 0-99"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Pre-encode prompts
if [ "$STEP" = "both" ] || [ "$STEP" = "encode" ]; then
    echo "========================================="
    echo "Step 1: Pre-encoding prompts with T5"
    echo "========================================="
    python /fs/nexus-projects/scene_graph_sd/DGE-T2I/src/models/flux_encode_prompts.py\
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$EMBEDDINGS_DIR" \
        --max_length 512
    echo ""
    echo "Encoding complete!"
    echo ""
fi

# Step 2: Generate images from embeddings
if [ "$STEP" = "both" ] || [ "$STEP" = "generate" ]; then
    echo "========================================="
    echo "Step 2: Generating images from embeddings"
    echo "========================================="
    
    CMD="python /fs/nexus-projects/scene_graph_sd/DGE-T2I/src/models/flux_generate_from_embeddings.py \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --embeddings_dir $EMBEDDINGS_DIR \
        --images_dir $IMAGES_DIR \
        --num_generations $NUM_GENERATIONS \
        --seed $SEED \
        --start_idx $START_IDX"
    
    if [ -n "$END_IDX" ]; then
        CMD="$CMD --end_idx $END_IDX"
    fi
    
    eval $CMD
    echo ""
    echo "Generation complete!"
    echo ""
fi

echo "========================================="
echo "Pipeline complete!"
echo "Embeddings: $EMBEDDINGS_DIR"
echo "Images: $IMAGES_DIR"
echo "========================================="

