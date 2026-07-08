#!/bin/bash

EPOCH=20
LR=1e-4
BATCHSIZE=2
TRAINSIZE=512
CLIP=0.5
DECAY_RATE=0.1
DECAY_EPOCH=80
LOAD=""
GPU_ID="0"
TRAIN_ROOT="/home/zj/Experiment/IML-DS/train/"
VAL_ROOT="/home/zj/Experiment/IML-DS/test/"
MODEL="train"
SAVE_PATH="/home/zj/Experiment/Other/PVT-RUN-IML/242-RUN-IML/RUN-IML/PVT-消融/xiao-5th-7.3-1"
LOG_PATH="/home/zj/Experiment/Other/PVT-RUN-IML/242-RUN-IML/RUN-IML/PVT-消融/log/xiao-5th-7.3-1.log"

mkdir -p /home/zj/Experiment/Other/PVT-RUN-IML/242-RUN-IML/RUN-IML/PVT-消融/log
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