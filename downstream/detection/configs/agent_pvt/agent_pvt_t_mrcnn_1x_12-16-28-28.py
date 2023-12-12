_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
pretrained = './data/agent_pvt_t_max_acc.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
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
        drop_path_rate=0.1, 
        depths=[2, 2, 2, 2], 
        sr_ratios=[8, 4, 2, 1], 
        agent_sr_ratios='1111', 
        num_stages=4,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes=[(12, 12), (16, 16), (28, 28), (28, 28)],
        kernel_size=3, 
        attn_type='AAAA',
        scale=-0.5,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(in_channels=[64, 128, 320, 512]))

lr = 2e-4

optimizer = dict(_delete_=True, type='AdamW', lr=lr, weight_decay=0.0001)

lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)