#!/bin/bash

EPOCH=20
LR=1e-4
BATCHSIZE=2
TRAINSIZE=512
CLIP=0.5
DECAY_RATE=0.1
LOAD=""
GPU_ID="0"
TRAIN_ROOT=""
VAL_ROOT=""
MODEL="train"
SAVE_PATH=""
LOG_PATH=""

mkdir -p 
mkdir -p ${SAVE_PATH}

CMD="python -u Train.py \
    --epoch ${EPOCH} \
    --lr ${LR} \
    --batchsize ${BATCHSIZE} \
    --trainsize ${TRAINSIZE} \
    --clip ${CLIP} \
    --decay_rate ${DECAY_RATE} \
    --decay_epoch ${DECAY_EPOCH} \
    --gpu_id ${GPU_ID} \
    --train_root ${TRAIN_ROOT} \
    --val_root ${VAL_ROOT} \
    --model ${MODEL} \
    --save_path ${SAVE_PATH}"

if [ -n "${LOAD}" ]; then
    CMD="${CMD} --load ${LOAD}"
fi

nohup ${CMD} > ${LOG_PATH} 2>&1 &

echo "Training started in background."
echo "Log file: ${LOG_PATH}"
echo "Process ID: $!"
