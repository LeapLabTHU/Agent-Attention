_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
checkpoint_file = './data/agent_swin_t_max_acc.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='AgentSwinTransformer',
        img_size=224, 
        patch_size=4, 
        in_chans=3,
        num_classes=80,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=56,
        mlp_ratio=4,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes = [(9, 9), (12, 12), (14, 14), (7, 7)],
        kernel_size=3, 
        attn_type='AAAB',
        scale=-0.5,
        ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'an_bias': dict(decay_mult=0.),
            'na_bias': dict(decay_mult=0.),
            'ah_bias': dict(decay_mult=0.),
            'aw_bias': dict(decay_mult=0.),
            'ha_bias': dict(decay_mult=0.),
            'wa_bias': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
