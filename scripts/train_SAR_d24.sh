#!/bin/bash
# SAR Training Script with CSFL - Depth 24
# In this example, we initialize from FlexVAR weights and train SAR with 10 epoches and disable sampling during student forcing for faster training.

# SAR d24 with CSFL training mode
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$node_rank --master_addr=$master_addr \
    train.py \
    --data_path=/path/to/imagenet \
    --exp_name SAR-student-forcing-CSFL-d24 \
    --vae_ckpt pretrained/FlexVAE.pth \
    --continue_training_ckpt pretrained/FlexVARd24-epo349.pth \
    --depth=24 --bs=768 --ep=360 --tblr=8e-5 --fp16=1 --alng=1e-4 --wpe=0.01 \
    --pn 1_2_3_4_5_6_8_10_13_16 \
    --training_mode csfl \
    --sigma 0.01 \
    --sf_use_sampling false \
    --vis_num_samples 32 \
    --vis_cfg_scale 2.5 \
    --vis_cosine_steps 1 \
    --vis_intermediate_scales 1 \
    "$@"
