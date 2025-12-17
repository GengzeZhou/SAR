#!/bin/bash
# SAR Training Script with CSFL - Depth 16
# In this example, we initialize from FlexVAR weights and train SAR with 10 epoches

# SAR d16 with CSFL training mode
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$node_rank --master_addr=$master_addr \
    train.py \
    --data_path=/path/to/imagenet \
    --exp_name SAR-student-forcing-CSFL-d16 \
    --vae_ckpt pretrained/FlexVAE.pth \
    --continue_training_ckpt pretrained/FlexVARd16-epo179.pth \
    --depth=16 --bs=768 --ep=190 --fp16=1 --alng=1e-3 --wpe=0.1 \
    --pn 1_2_3_4_5_6_8_10_13_16 \
    --training_mode csfl \
    --sigma 0.01 \
    --sf_use_sampling 1 \
    --sf_cfg_scale 2.5 \
    --sf_top_k 900 \
    --sf_top_p 0.96 \
    --vis_num_samples 32 \
    --vis_cfg_scale 2.5 \
    --vis_cosine_steps 1 \
    --vis_intermediate_scales 1 \
    "$@"
