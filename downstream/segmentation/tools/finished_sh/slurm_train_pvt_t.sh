# #!/usr/bin/env bash
set -x

PARTITION=RTX3090
JOB_NAME=pvt_t_seg
CONFIG=configs/slide_pvt/fpn_slide_pvt_t.py
GPUS=8
GPUS_PER_NODE=8
# GPUS=1
# GPUS_PER_NODE=1
CPUS_PER_GPU=8
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --cpus-per-gpu=${CPUS_PER_GPU} \
    --kill-on-bad-exit=1 \
    -x node06,node07 \
    ${SRUN_ARGS} \
    torchrun --nproc_per_node 8 tools/train.py ${CONFIG} --launcher="pytorch" \
    --load-from './data/pvt_t_model.pth'