# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .agent_swin import AgentSwinTransformer
from .agent_deit import agent_deit_tiny, agent_deit_small, agent_deit_base
from .agent_pvt import agent_pvt_tiny, agent_pvt_small, agent_pvt_medium, agent_pvt_large
from .agent_cswin import Agent_CSWin_64_24181_tiny_224, Agent_CSWin_96_36292_base_224, \
    Agent_CSWin_96_36292_base_384, Agent_CSWin_64_36292_small_224


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'agent_swin':
        model = AgentSwinTransformer(img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                     in_chans=config.MODEL.SWIN.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                     depths=config.MODEL.SWIN.DEPTHS,
                                     num_heads=config.MODEL.SWIN.NUM_HEADS,
                                     window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                     qk_scale=config.MODEL.SWIN.QK_SCALE,
                                     drop_rate=config.MODEL.DROP_RATE,
                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                     ape=config.MODEL.SWIN.APE,
                                     patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                     agent_num=config.MODEL.AGENT.NUM.split('-'),
                                     attn_type=config.MODEL.AGENT.ATTN_TYPE)

    elif model_type in ['agent_deit_tiny', 'agent_deit_small', 'agent_deit_base',
                        'agent_deit_base_d21', 'agent_deit_mini_2x', 'agent_deit_tiny_2x',
                        'agent_deit_tiny_4x', 'agent_deit_small_2x',
                        'agent_deit_base_agent2', 'agent_deit_base_agent4', 'agent_deit_base_agent6']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'agent_num=config.MODEL.AGENT.NUM.split(\'-\'))')

    elif model_type in ['agent_pvt_tiny', 'agent_pvt_small', 'agent_pvt_medium', 'agent_pvt_large',
                        'agent_pvt_tiny_2x']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'agent_num=config.MODEL.AGENT.NUM.split(\'-\'),'
                                  'attn_type=config.MODEL.AGENT.ATTN_TYPE,'
                                  'agent_sr_ratios=str(config.MODEL.AGENT.PVT_LA_SR_RATIOS))')

    elif model_type in ['Agent_CSWin_64_24181_tiny_224', 'Agent_CSWin_64_24322_small_224',
                        'Agent_CSWin_96_36292_base_224', 'Agent_CSWin_64_36292_small_224',
                        'Agent_CSWin_96_36292_base_384']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'agent_num=config.MODEL.AGENT.NUM.split(\'-\'),'
                                  'attn_type=config.MODEL.AGENT.ATTN_TYPE,'
                                  'la_split_size=config.MODEL.AGENT.CSWIN_LA_SPLIT_SIZE)')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
