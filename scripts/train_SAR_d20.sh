#!/bin/bash
# SAR Training Script with CSFL - Depth 20
# In this example, we initialize from FlexVAR weights and train SAR with 10 epoches and disable sampling during student forcing for faster training.

# SAR d20 with CSFL training mode
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$node_rank --master_addr=$master_addr \
    train.py \
    --data_path=/path/to/imagenet \
    --exp_name SAR-student-forcing-CSFL-d20 \
    --vae_ckpt pretrained/FlexVAE.pth \
    --continue_training_ckpt pretrained/FlexVARd20-epo249.pth \
    --depth=20 --bs=768 --ep=260 --fp16=1 --alng=1e-3 --wpe=0.1 \
    --pn 1_2_3_4_5_6_8_10_13_16 \
    --training_mode csfl \
    --sigma 0.01 \
    --sf_use_sampling false \
    --vis_num_samples 32 \
    --vis_cfg_scale 2.5 \
    --vis_cosine_steps 1 \
    --vis_intermediate_scales 1 \
    "$@"
