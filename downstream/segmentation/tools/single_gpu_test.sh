#!/usr/bin/env bash

set -x

# --ntasks=${GPUS} \

PARTITION=RTX3090
JOB_NAME=seg_t
CONFIG=configs/agent_swin/upernet_agent_swin_t.py
# GPUS=8
# GPUS_PER_NODE=8
GPUS=1
GPUS_PER_NODE=1
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    torchrun --nproc_per_node 1 tools/train.py ${CONFIG} --launcher="pytorch"