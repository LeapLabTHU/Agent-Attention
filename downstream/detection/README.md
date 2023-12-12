# Agent Attention for Object Detection

Code and configuration files to reproduce object detection results of our paper. All experiments are conducted on COCO datast based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-Swin-T | ImageNet-1K | 1x | 44.6 | 40.7 | 48M | 276G | [config](configs/agent_swin/agent_swin_t_mrcnn_1x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/bc3077b2c1164dcf9ef2/?dl=1) |
| Agent-Swin-T | ImageNet-1K | 3x | 47.3 | 42.7 | 48M | 276G | [config](configs/agent_swin/agent_swin_t_mrcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/46b3a73aa52b4ee29c47/?dl=1) |
| Agent-Swin-S | ImageNet-1K | 1x | 47.2 | 42.7 | 69M | 364G | [config](configs/agent_swin/agent_swin_s_mrcnn_1x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/6675c65e4bc24bffb2aa/?dl=1) |
| Agent-Swin-S | ImageNet-1K | 3x | 48.9 | 43.8 | 69M | 364G | [config](configs/agent_swin/agent_swin_s_mrcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/6b6048b2d04c44a09f16/?dl=1) |
| Agent-PVT-T | ImageNet-1K | 1x | 41.4 | 38.7 | 31M | 230G | [config](configs/agent_pvt/agent_pvt_t_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/a6037e90e08040a189c5/?dl=1) |
| Agent-PVT-S | ImageNet-1K | 1x | 44.5 | 41.2 | 40M | 293G | [config](configs/agent_pvt/agent_pvt_s_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/27e657035a684607bca8/?dl=1) |
| Agent-PVT-M | ImageNet-1K | 1x | 45.9 | 42.0 | 56M | 400G | [config](configs/agent_pvt/agent_pvt_m_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/ff740ea3feed43e0ada1/?dl=1) |
| Agent-PVT-L | ImageNet-1K | 1x | 46.9 | 42.8 | 68M | 510G | [config](configs/agent_pvt/agent_pvt_l_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/63c443f60f6047e79985/?dl=1) |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-Swin-T | ImageNet-1K | 1x | 49.2 | 42.7 | 86M | 755G | [config](configs/agent_swin/agent_swin_t_crcnn_1x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/0fbffcc93877460ab5d2/?dl=1) |
| Agent-Swin-T | ImageNet-1K | 3x | 51.4 | 44.5 | 86M | 755G | [config](configs/agent_swin/agent_swin_t_crcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/f82a6a4beed74f81a5b5/?dl=1) |
| Agent-Swin-S | ImageNet-1K | 3x | 52.6 | 45.5 | 107M | 843G | [config](configs/agent_swin/agent_swin_s_crcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/ed71c939c6c6441ea496/?dl=1) |
| Agent-Swin-B | ImageNet-1K | 3x | 52.6 | 45.3 | 145M | 990G | [config](configs/agent_swin/agent_swin_b_mrcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/90def6531521437b8c79/?dl=1) |

### RetinaNet

| Backbone | Pretrain | Lr Schd | box mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-PVT-T | ImageNet-1K | 1x | 40.3 | 21M | 211G | [config](configs/agent_pvt/agent_pvt_t_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3b26e1b0498f4819a18d/?dl=1) |
| Agent-PVT-S | ImageNet-1K | 1x | 44.1 | 30M | 274G | [config](configs/agent_pvt/agent_pvt_s_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/76ee78f9ed0e4041b1d8/?dl=1) |
| Agent-PVT-M | ImageNet-1K | 1x | 45.8 | 46M | 382G | [config](configs/agent_pvt/agent_pvt_m_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/6a0daf14bdda447683c9/?dl=1) |
| Agent-PVT-L | ImageNet-1K | 1x | 46.8 | 58M | 492G | [config](configs/agent_pvt/agent_pvt_l_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/eedb9d74735140989df9/?dl=1) |

## Usage

### Dataset

Prepare COCO dataset, and change `data_root` argument in `configs/_base_/datasets/coco_detection.py` and `configs/_base_/datasets/coco_instance.py` to the dataset path.

### ImageNet-1K Pretrained Model

Please place ImageNet-1K pretrained models under `./data/` folder and rename them as `{MODEL_STRUCTURE}_max_acc.pth`, e.g. `agent_swin_t_max_acc.pth`.

### Installation

For convenience, we provide the conda environment file.
```
conda env create -f agent_detection.yaml
cd ../mmcv/
pip install -v -e .
cd ../detection/
pip install -v -e .
```

### Inference

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE>

# multi-gpu training
torchrun --nproc_per_node <GPU_NUM> tools/train.py <CONFIG_FILE> --launcher="pytorch"
```

## Citation

If you find this repo helpful, please consider citing us.

```latex
(To be updated)
```

