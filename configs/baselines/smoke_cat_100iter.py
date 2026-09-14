_base_ = '../../third_party/mmdetection-3.3.0/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py'

custom_imports = dict(imports=['research.mmdet_plugins'], allow_failed_imports=False)
load_from = 'weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
model = dict(
    language_model=dict(
        name='weights/bert-base-uncased',
        use_checkpoint=True),
    backbone=dict(with_cp=True),
    encoder=dict(num_cp=6))

# Smoke-only resolution fallback after batch=1, accumulation and AMP. It is
# not a benchmark metric and does not alter train/test membership.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(640, 1067), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixScaleResize', scale=(640, 1067), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale=1.0,
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.0),
        }))
custom_hooks = [
    dict(type='SetTrainableModulesHook', freeze_prefixes=['language_model']),
    dict(type='GradientAndMemoryAuditHook', interval=10)
]
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=100,
    val_interval=100)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=False,
        milestones=[70],
        gamma=0.1)
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=1),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,
        interval=1,
        score_thr=0.25,
        test_out_dir='visualizations'))
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)
randomness = dict(seed=42, deterministic=True)
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = 'experiments/smoke_cat_100iter'
