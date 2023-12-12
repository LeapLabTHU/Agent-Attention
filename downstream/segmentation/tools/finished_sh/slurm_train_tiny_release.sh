# #!/usr/bin/env bash

# set -x

# PARTITION=RTX3090
# JOB_NAME=slide_transformer_tiny_seg
# CONFIG=configs/slide_transformer/upernet_slide_transformer_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
# GPUS=8
# GPUS_PER_NODE=8
# # GPUS=1
# # GPUS_PER_NODE=1
# SRUN_ARGS=${SRUN_ARGS:-""}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --kill-on-bad-exit=1 \
#     -x node06,node07,node09 \
#     ${SRUN_ARGS} \
#     python -u tools/train.py ${CONFIG} --launcher="slurm" \
#     --load-from './data/ckpt_epoch_300_model.pth'


#!/usr/bin/env bash

set -x

# --ntasks=${GPUS} \

PARTITION=RTX3090
JOB_NAME=seg_t_rel
CONFIG=configs/slide_transformer/upernet_slide_transformer_release_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
GPUS=8
GPUS_PER_NODE=8
# GPUS=1
# GPUS_PER_NODE=1
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    -x node06,node07 \
    ${SRUN_ARGS} \
    torchrun --nproc_per_node 8 tools/train.py ${CONFIG} --launcher="pytorch" \
    --load-from './data/ckpt_epoch_300_model_release.pth'
