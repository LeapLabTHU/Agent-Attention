# Agent Transformer

This folder contains the implementation of Agent Attention based on DeiT, PVT, Swin and CSwin models for image classification.

## Dependencies

- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy
- timm == 0.4.12
- einops
- yacs

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Pretrained Models

| model  | Reso | Params | FLOPs | acc@1 | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent-DeiT-T | $224^2$ | 6.0M | 1.2G | 74.9 | [config](cfgs/agent_deit_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/57e0904221824ee7b7af/?dl=1) |
| Agent-DeiT-S | $224^2$ | 22.7M | 4.4G | 80.5 | [config](cfgs/agent_deit_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/fdd43304372541db8ea7/?dl=1) |
| Agent-DeiT-S | $448^2$ | 23.1M  | 17.7G | 83.1  | [config](cfgs/agent_deit_s_448.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/a228f86241c445f1969b/?dl=1) |
| Agent-DeiT-B | $224^2$ | 87.2M | 17.6G | 82.0 | [config](cfgs/agent_deit_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/5115f15fdd7e4ab58fbc/?dl=1) |
| Agent-PVT-T | $224^2$ | 11.6M | 2.0G | 78.4 | [config](cfgs/agent_pvt_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/13fdf0e0a7c5498c934f/?dl=1) |
| Agent-PVT-S | $224^2$ | 20.6M | 4.0G | 82.2 | [config](cfgs/agent_pvt_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/36b8be9a8863456685a3/?dl=1) |
| Agent-PVT-M | $224^2$ | 35.9M | 7.0G | 83.4 | [config](cfgs/agent_pvt_m.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/4d00ecacf18b45f9b88c/?dl=1) |
| Agent-PVT-M | $256^2$ | 36.1M | 9.2G | 83.8 | [config](cfgs/agent_pvt_m_256.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/654df734121643b19c52/?dl=1) |
| Agent-PVT-L | $224^2$ | 48.7M | 10.4G | 83.7 | [config](cfgs/agent_pvt_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/040cfe0cfbbe4762ad09/?dl=1) |
| Agent-Swin-T | $224^2$ | 29M | 4.5G | 82.6 | [config](cfgs/agent_swin_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/a2ddc1190c224f768a41/?dl=1) |
| Agent-Swin-S | $224^2$ | 50M | 8.7G | 83.7 | [config](cfgs/agent_swin_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/f607172c9fea4d6e99bb/?dl=1) |
| Agent-Swin-S | $288^2$ | 50M | 14.6G | 84.1 | [config](cfgs/agent_swin_s_288.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/09760394e93b4046a25e/?dl=1) |
| Agent-Swin-B | $224^2$ | 88M | 15.4G | 84.0 | [config](cfgs/agent_swin_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/4fcd2d0114bb490483c3/?dl=1) |
| Agent-Swin-B | $384^2$ | 88M | 46.3G | 84.9 | [config](cfgs/agent_swin_b_384.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/bb84ee9058954706a557/?dl=1) |
| Agent-CSwin-T | $224^2$ | 21M | 4.3G | 83.1 | [config](cfgs/agent_cswin_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8c562a74441e46d5902b/?dl=1) |
| Agent-CSwin-S | $224^2$ | 33M | 6.8G | 83.9 | [config](cfgs/agent_cswin_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/30eb8af675bc4b7a8747/?dl=1) |
| Agent-CSwin-B | $224^2$ | 73M | 14.9G | 84.7 | [config](cfgs/agent_cswin_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/c8035b0839b2410c8c9f/?dl=1) |
| Agent-CSwin-B | $384^2$ | 73M | 46.3G | 85.8 | [config](cfgs/agent_cswin_b_384.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/1239be0df44b47719432/?dl=1) |

Evaluate `Agent-DeiT/Agent-PVT/Agent-Swin` on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

Evaluate `Agent-CSwin` on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

## Train Models from Scratch

- To train `Agent-DeiT/Agent-PVT/Agent-Swin` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path>
```

- To train `Agent-CSwin-T/S/B` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --model-ema --model-ema-decay 0.99984/0.99984/0.99992
```

## Fine-tuning on higher resolution

- Fine-tune a `Agent-Swin-B` model pre-trained on 224x224 resolution to 384x384 resolution:


```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/agent_swin_b_384.yaml --data-path <imagenet-path> --output <output-path> --pretrained <path-to-224x224-pretrained-weights>
```

- Fine-tune a `Agent-CSwin-B` model pre-trained on 224x224 resolution to 384x384 resolution:


```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg ./cfgs/agent_cswin_b_384.yaml --data-path <imagenet-path> --output <output-path> --pretrained <path-to-224x224-pretrained-weights> --model-ema --model-ema-decay 0.9998
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
