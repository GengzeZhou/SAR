#!/bin/bash
# SAR Evaluation Script - Depth 16

set -e

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
echo "Using ${NUM_GPUS} GPUs"

# Configuration
VAR_CKPT="pretrained/SARd16-epo179.pth"
VAE_CKPT="pretrained/FlexVAE.pth"
MODEL_DEPTH=16
CFG=2.2
NUM_SAMPLES=50
BATCH_SIZE=128

# Create output directory
OUTPUT_DIR="evaluation_results/sar_d${MODEL_DEPTH}_cfg${CFG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

echo "======================================"
echo "SAR Evaluation - Depth ${MODEL_DEPTH}"
echo "======================================"
echo "Configuration:"
echo "  Checkpoint: ${VAR_CKPT}"
echo "  CFG Scale: ${CFG}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "======================================"

torchrun --standalone --nproc_per_node=${NUM_GPUS} eval_c2i.py \
    --var_ckpt ${VAR_CKPT} \
    --vae_ckpt ${VAE_CKPT} \
    --depth ${MODEL_DEPTH} \
    --cfg ${CFG} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_path ${OUTPUT_DIR} \
    --results_path ${OUTPUT_DIR}

echo "Results saved to: ${OUTPUT_DIR}"
