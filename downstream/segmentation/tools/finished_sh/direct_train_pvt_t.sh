# #!/usr/bin/env bash
torchrun --nproc_per_node 8 --master_port=25700 tools/train.py configs/slide_pvt/fpn_slide_pvt_t.py \
    --launcher="pytorch" --load-from './data/pvt_t_model.pth'