# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math

import numpy as np
import torch

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor
from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string



def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 2048],
        help='input image size')
    args = parser.parse_args()
    return args


def sra_flops(h, w, r, dim, stage):
    print('shape', h, w, r)
    print('dim', dim)
    print('stage', stage)
    if not stage:
        return 2 * h * w * (h // r) * (w // r) * dim
    else:
        return 2 * h * w * 3 * 3 * dim

def agentv5_sra_flops(h, w, r, dim, group_size, downstream_agent_shape, attn_type):
    print('shape', h, w, r)
    print('dim', dim)
    print('group_size', group_size)
    print('downstream_shape', downstream_agent_shape)
    print('attn_type', attn_type)
    if attn_type == 'A':
        return 2 * h * w * (h // r) * (w // r) * dim
    else:
        return 4 * h * w * downstream_agent_shape[0] * downstream_agent_shape[1] * dim

def wmsa_flops(h, w, dim, stage):
    print('shape', h, w)
    print('dim', dim)
    print('stage', stage)
    if not stage:
        return 2 * h * w * 7 * 7 * dim
    else:
        return 2 * h * w * 3 * 3 * dim

def wmsa_ad_flops(h, w, dim, group_size, attn_type):
    print('shape', h, w)
    print('dim', dim)
    print('group_size', group_size)
    print('attn_type', attn_type)
    if attn_type == 'A':
        return 2 * h * w * 7 * 7 * dim
    elif attn_type == 'B':
        return 4 * h * w * group_size * dim

def wmsa_adv4_flops(h, w, cls_reso, dim, group_size, down_stream_scale_factor, attn_type):
    print('shape', h, w)
    print('cls reso', cls_reso)
    print('dim', dim)
    print('group_size', group_size)
    print('down_stream_scale_factor', down_stream_scale_factor)
    print('attn_type', attn_type)
    if attn_type == 'A':
        return 2 * h * w * 7 * 7 * dim
    elif attn_type == 'B':
        pool_size = int(group_size ** 0.5)
        return 4 * h * w * (math.ceil(h / cls_reso) // down_stream_scale_factor) * pool_size * \
            (math.ceil(w / cls_reso) // down_stream_scale_factor) * pool_size * dim

def wmsa_adv5_flops(h, w, dim, group_size, downstream_agent_shape, attn_type):
    print('shape', h, w)
    print('dim', dim)
    print('group_size', group_size)
    print('attn_type', attn_type)
    print('downstream_agent_shape', downstream_agent_shape)
    if attn_type == 'A':
        return 2 * h * w * 7 * 7 * dim
    elif attn_type == 'B':
        return 4 * h * w * downstream_agent_shape[0] * downstream_agent_shape[1] * dim

def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)

    backbone = model.backbone
    backbone_name = type(backbone).__name__
    _, H, W = input_shape
    if 'Pyramid' in backbone_name:
        # 这里按照每个stage里纯由某种block组成来算的
        stage1 = sra_flops(H // 4, W // 4,
                               backbone.block1[0].attn.sr_ratio,
                               backbone.block1[0].attn.dim, backbone.stages[0]) * len(backbone.block1)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = sra_flops(H // 8, W // 8,
                               backbone.block2[0].attn.sr_ratio,
                               backbone.block2[0].attn.dim, backbone.stages[1]) * len(backbone.block2)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = sra_flops(H // 16, W // 16,
                               backbone.block3[0].attn.sr_ratio,
                               backbone.block3[0].attn.dim, backbone.stages[2]) * len(backbone.block3)
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = sra_flops(H // 32, W // 32,
                               backbone.block4[0].attn.sr_ratio,
                               backbone.block4[0].attn.dim, backbone.stages[3]) * len(backbone.block4)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')

    elif 'SlideTransformer' in backbone_name:
        # 这里按照每个stage里纯由某种block组成来算的
        stage1 = wmsa_flops(H // 4, W // 4,
                               backbone.layers[0].blocks[0].attn.dim, backbone.stages[0]) * len(backbone.layers[0].blocks)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = wmsa_flops(H // 8, W // 8,
                               backbone.layers[1].blocks[0].attn.dim, backbone.stages[1]) * len(backbone.layers[1].blocks)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = wmsa_flops(H // 16, W // 16,
                               backbone.layers[2].blocks[0].attn.dim, backbone.stages[2]) * len(backbone.layers[2].blocks)
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = wmsa_flops(H // 32, W // 32,
                               backbone.layers[3].blocks[0].attn.dim, backbone.stages[3]) * len(backbone.layers[3].blocks)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')

    elif backbone_name == 'SwinTransformerAD':
        # 这里按照每个stage里纯由某种block组成来算的
        # 注意这里默认都是第一个stage h w // 4，后边每两个stage之间 // 2
        stage1 = wmsa_ad_flops(H // 4, W // 4, 
                               backbone.layers[0].blocks[0].attn.dim, backbone.group_sizes[0], backbone.attn_type[0]) * len(backbone.layers[0].blocks)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = wmsa_ad_flops(H // 8, W // 8,
                               backbone.layers[1].blocks[0].attn.dim, backbone.group_sizes[1], backbone.attn_type[1]) * len(backbone.layers[1].blocks)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = wmsa_ad_flops(H // 16, W // 16,
                               backbone.layers[2].blocks[0].attn.dim, backbone.group_sizes[2], backbone.attn_type[2]) * len(backbone.layers[2].blocks)
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = wmsa_ad_flops(H // 32, W // 32,
                               backbone.layers[3].blocks[0].attn.dim, backbone.group_sizes[3], backbone.attn_type[3]) * len(backbone.layers[3].blocks)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')

    elif backbone_name == 'SwinTransformerADV4':
        # 这里按照每个stage里纯由某种block组成来算的
        # 注意这里默认都是第一个stage h w // 4，后边每两个stage之间 // 2
        stage1 = wmsa_adv4_flops(H // 4, W // 4,
                                56,
                                backbone.layers[0].blocks[0].attn.dim, 
                                backbone.group_sizes[0],
                                backbone.down_stream_scale_factors[0],
                                backbone.attn_type[0]) * len(backbone.layers[0].blocks)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = wmsa_adv4_flops(H // 8, W // 8,
                                28,
                                backbone.layers[1].blocks[0].attn.dim, 
                                backbone.group_sizes[1], 
                                backbone.down_stream_scale_factors[1],
                                backbone.attn_type[1]) * len(backbone.layers[1].blocks)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = wmsa_adv4_flops(H // 16, W // 16,
                                14,
                                backbone.layers[2].blocks[0].attn.dim, 
                                backbone.group_sizes[2], 
                                backbone.down_stream_scale_factors[2],
                                backbone.attn_type[2]) * len(backbone.layers[2].blocks)
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = wmsa_adv4_flops(H // 32, W // 32,
                                7,
                                backbone.layers[3].blocks[0].attn.dim, 
                                backbone.group_sizes[3],
                                backbone.down_stream_scale_factors[3],
                                backbone.attn_type[3]) * len(backbone.layers[3].blocks)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')

    elif backbone_name == 'SwinTransformerADV5':
        # 这里按照每个stage里纯由某种block组成来算的
        # 注意这里默认都是第一个stage h w // 4，后边每两个stage之间 // 2
        stage1 = wmsa_adv5_flops(H // 4, W // 4, 
                               backbone.layers[0].blocks[0].attn.dim, 
                               backbone.group_sizes[0],
                               backbone.downstream_agent_shapes[0],
                               backbone.attn_type[0]) * len(backbone.layers[0].blocks)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = wmsa_adv5_flops(H // 8, W // 8,
                               backbone.layers[1].blocks[0].attn.dim, 
                               backbone.group_sizes[1], 
                               backbone.downstream_agent_shapes[1],
                               backbone.attn_type[1]) * len(backbone.layers[1].blocks)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = wmsa_adv5_flops(H // 16, W // 16,
                               backbone.layers[2].blocks[0].attn.dim, 
                               backbone.group_sizes[2], 
                               backbone.downstream_agent_shapes[2],
                               backbone.attn_type[2]) * len(backbone.layers[2].blocks)
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = wmsa_adv5_flops(H // 32, W // 32,
                               backbone.layers[3].blocks[0].attn.dim, 
                               backbone.group_sizes[3],
                               backbone.downstream_agent_shapes[3],
                               backbone.attn_type[3]) * len(backbone.layers[3].blocks)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')
    
    elif backbone_name == 'AgentSwinTransformerV5':
        stage1 = wmsa_adv5_flops(H // 4, W // 4, 
                               backbone.layers[0].blocks[0].attn.dim, 
                               backbone.agent_num[0],
                               backbone.downstream_agent_shapes[0],
                               'B') * len(backbone.layers[0].blocks)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = wmsa_adv5_flops(H // 8, W // 8,
                               backbone.layers[1].blocks[0].attn.dim, 
                               backbone.agent_num[1], 
                               backbone.downstream_agent_shapes[1],
                               'B') * len(backbone.layers[1].blocks)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = wmsa_adv5_flops(H // 16, W // 16,
                               backbone.layers[2].blocks[0].attn.dim, 
                               backbone.agent_num[2], 
                               backbone.downstream_agent_shapes[2],
                               'B') * 2 + \
                 wmsa_adv5_flops(H // 16, W // 16,
                               backbone.layers[2].blocks[0].attn.dim, 
                               backbone.agent_num[2], 
                               backbone.downstream_agent_shapes[2],
                               'A') * 16
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = wmsa_adv5_flops(H // 32, W // 32,
                               backbone.layers[3].blocks[0].attn.dim, 
                               backbone.agent_num[3],
                               backbone.downstream_agent_shapes[3],
                               'A') * len(backbone.layers[3].blocks)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')
    
    elif backbone_name == 'AgentPVTV5':
        # 这里按照每个stage里纯由某种block组成来算的
        # 注意这里默认都是第一个stage h w // 4，后边每两个stage之间 // 2
        stage1 = agentv5_sra_flops(H // 4, W // 4,
                                backbone.block1[0].attn.sr_ratio,
                                backbone.block1[0].attn.dim, 
                                backbone.agent_num[0],
                                backbone.downstream_agent_shapes[0],
                                backbone.attn_type[0]) * len(backbone.block1)
        print('stage1 flops:', stage1 / 1e9, 'G')
        stage2 = agentv5_sra_flops(H // 8, W // 8,
                                backbone.block2[0].attn.sr_ratio,
                                backbone.block2[0].attn.dim, 
                                backbone.agent_num[1],
                                backbone.downstream_agent_shapes[1],
                                backbone.attn_type[1]) * len(backbone.block2)
        print('stage2 flops:', stage2 / 1e9, 'G')
        stage3 = agentv5_sra_flops(H // 16, W // 16,
                                backbone.block3[0].attn.sr_ratio,
                                backbone.block3[0].attn.dim, 
                                backbone.agent_num[2],
                                backbone.downstream_agent_shapes[2],
                                backbone.attn_type[2]) * len(backbone.block3)
        print('stage3 flops:', stage3 / 1e9, 'G')
        stage4 = agentv5_sra_flops(H // 32, W // 32,
                                backbone.block4[0].attn.sr_ratio,
                                backbone.block4[0].attn.dim, 
                                backbone.agent_num[3],
                                backbone.downstream_agent_shapes[3],
                                backbone.attn_type[3]) * len(backbone.block4)
        print('stage4 flops:', stage4 / 1e9, 'G')
        attn_flops = stage1 + stage2 + stage3 + stage4
        flops += attn_flops
        print('Attn flops: ', attn_flops / 1e9, 'G')
    else:
        print('Not Swin/PVT model!')

    # if 'Pyramid' in backbone_name and backbone.stages==[False, False, False, False]:
    #     stage1 = sra_flops(H // 4, W // 4, 
    #                         backbone.layers[0][1][1].attn.sr_ratio, 
    #                         backbone.layers[0][1][1].attn.embed_dims) * backbone.num_layers[0]
    #     stage2 = sra_flops(H // 8, W // 8, 
    #                         backbone.layers[1][1][1].attn.sr_ratio, 
    #                         backbone.layers[1][1][1].attn.embed_dims) * backbone.num_layers[1]
    #     stage3 = sra_flops(H // 16, W // 16, 
    #                         backbone.layers[2][1][1].attn.sr_ratio, 
    #                         backbone.layers[2][1][1].attn.embed_dims) * backbone.num_layers[2]
    #     stage4 = sra_flops(H // 32, W // 32, 
    #                         backbone.layers[3][1][1].attn.sr_ratio, 
    #                         backbone.layers[3][1][1].attn.embed_dims) * backbone.num_layers[3]
    #     attn_flops = stage1 + stage2 + stage3 + stage4
    #     flops += attn_flops
    #     print('Attn flops: ', attn_flops)
    
    return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_flops(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
