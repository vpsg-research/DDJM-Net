#!/bin/bash

# Training script for Train_pvt.py with nohup support

# Default parameters (can be modified as needed)
EPOCH=300
LR=1e-4
BATCHSIZE=4
TRAINSIZE=512
CLIP=0.5
DECAY_RATE=0.1
DECAY_EPOCH=80
LOAD="model/pretrained/base_patch16_384.pth"
GPU_ID="1"
TRAIN_ROOT="../IML-DS/train/"
VAL_ROOT="../IML-DS/test/"
MODEL="train"
SAVE_PATH="log/12.21_xiao_zhi-0.15/"

# Run training with nohup in background
nohup python -u Train.py \
    --epoch ${EPOCH} \
    --lr ${LR} \
    --batchsize ${BATCHSIZE} \
    --trainsize ${TRAINSIZE} \
    --clip ${CLIP} \
    --decay_rate ${DECAY_RATE} \
    --decay_epoch ${DECAY_EPOCH} \
    --load ${LOAD} \
    --gpu_id ${GPU_ID} \
    --train_root ${TRAIN_ROOT} \
    --val_root ${VAL_ROOT} \
    --model ${MODEL} \
    --save_path ${SAVE_PATH} \
    > 日志/12.21_xiao_zhi-0.15.log 2>&1 &

echo "Training started in background. Check train.log for output."  
echo "Process ID: $!" 