# Agent Attention for Object Detection

Code and configuration files to reproduce object detection results of our paper. All experiments are conducted on COCO datast based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-Swin-T | ImageNet-1K | 1x | 44.6 | 40.7 | 48M | 276G | [config](configs/agent_swin/agent_swin_t_mrcnn_1x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/258e208469084e6abf26/?dl=1) |
| Agent-Swin-T | ImageNet-1K | 3x | 47.3 | 42.7 | 48M | 276G | [config](configs/agent_swin/agent_swin_t_mrcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/e0ab821875704960a505/?dl=1) |
| Agent-Swin-S | ImageNet-1K | 1x | 47.2 | 42.7 | 69M | 364G | [config](configs/agent_swin/agent_swin_s_mrcnn_1x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/52714305c21a4104bbc5/?dl=1) |
| Agent-Swin-S | ImageNet-1K | 3x | 48.9 | 43.8 | 69M | 364G | [config](configs/agent_swin/agent_swin_s_mrcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/b78f6ff837b84b61b122/?dl=1) |
| Agent-PVT-T | ImageNet-1K | 1x | 41.4 | 38.7 | 31M | 230G | [config](configs/agent_pvt/agent_pvt_t_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/79cbc3a7b7a24a0b92c1/?dl=1) |
| Agent-PVT-S | ImageNet-1K | 1x | 44.5 | 41.2 | 40M | 293G | [config](configs/agent_pvt/agent_pvt_s_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8e333aa6b55e45d4b76c/?dl=1) |
| Agent-PVT-M | ImageNet-1K | 1x | 45.9 | 42.0 | 56M | 400G | [config](configs/agent_pvt/agent_pvt_m_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/6401bf2b227046a58fc3/?dl=1) |
| Agent-PVT-L | ImageNet-1K | 1x | 46.9 | 42.8 | 68M | 510G | [config](configs/agent_pvt/agent_pvt_l_mrcnn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8fa3d506f1e24099b99f/?dl=1) |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-Swin-T | ImageNet-1K | 1x | 49.2 | 42.7 | 86M | 755G | [config](configs/agent_swin/agent_swin_t_crcnn_1x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/39d78fddf90241e58db0/?dl=1) |
| Agent-Swin-T | ImageNet-1K | 3x | 51.4 | 44.5 | 86M | 755G | [config](configs/agent_swin/agent_swin_t_crcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/ed2af911d0e64d359aa3/?dl=1) |
| Agent-Swin-S | ImageNet-1K | 3x | 52.6 | 45.5 | 107M | 843G | [config](configs/agent_swin/agent_swin_s_crcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3d092a23b935458b98ed/?dl=1) |
| Agent-Swin-B | ImageNet-1K | 3x | 52.6 | 45.3 | 145M | 990G | [config](configs/agent_swin/agent_swin_b_mrcnn_3x_9-12-14-7.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/b3a628e668eb460ca532/?dl=1) |

### RetinaNet

| Backbone | Pretrain | Lr Schd | box mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-PVT-T | ImageNet-1K | 1x | 40.3 | 21M | 211G | [config](configs/agent_pvt/agent_pvt_t_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/d3c1dbe0243d4775aafd/?dl=1) |
| Agent-PVT-S | ImageNet-1K | 1x | 44.1 | 30M | 274G | [config](configs/agent_pvt/agent_pvt_s_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/d427a52511f64ab5a928/?dl=1) |
| Agent-PVT-M | ImageNet-1K | 1x | 45.8 | 46M | 382G | [config](configs/agent_pvt/agent_pvt_m_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/81d6c824d9f244a192a3/?dl=1) |
| Agent-PVT-L | ImageNet-1K | 1x | 46.8 | 58M | 492G | [config](configs/agent_pvt/agent_pvt_l_rtn_1x_12-16-28-28.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/2dc5b1f685fd4ecf8db5/?dl=1) |

## Usage

### Dataset

Prepare COCO dataset, and change `data_root` argument in `configs/_base_/datasets/coco_detection.py` and `configs/_base_/datasets/coco_instance.py` to the dataset path.

### ImageNet-1K Pretrained Model

Please place ImageNet-1K pretrained models under `./data/` folder and rename them as `{MODEL_STRUCTURE}_max_acc.pth`, e.g. `agent_swin_t_max_acc.pth`.

### Installation

For convenience, we provide the conda environment file and pre-bulit `mmcv`.
Please download the pre-built mmcv [here](https://cloud.tsinghua.edu.cn/d/b9bb25fcdc49430c9d87/), and place it under `../` 
We use an empty `mmcv` directory as a placeholder.
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
@article{han2023agent,
  title={Agent Attention: On the Integration of Softmax and Linear Attention},
  author={Han, Dongchen and Ye, Tianzhu and Han, Yizeng and Xia, Zhuofan and Song, Shiji and Huang, Gao},
  journal={arXiv preprint arXiv:2312.08874},
  year={2023}
}
```

