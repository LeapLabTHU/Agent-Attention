_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint_file = './data/agent_pvt_m_max_acc.pth' 

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        _delete_=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='AgentPVT',
        img_size=224, 
        patch_size=4, 
        in_chans=3, 
        num_classes=80, 
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4], 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0.,
        attn_drop_rate=0., 
        drop_path_rate=0.2, 
        depths=[3, 4, 18, 3], 
        sr_ratios=[8, 4, 2, 1], 
        agent_sr_ratios='1111', 
        num_stages=4,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes = [(12, 12), (16, 16), (28, 28), (28, 28)],
        kernel_size=3, 
        attn_type='AAAA',
        scale=-0.5),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0002,
    weight_decay=0.0001)

lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
