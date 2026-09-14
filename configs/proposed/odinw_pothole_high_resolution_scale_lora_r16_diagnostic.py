_base_ = './odinw_pothole_scale_aware_lora_r16.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(960, 1600), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixScaleResize', scale=(960, 1600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        train_only_substrings=['lora_down', 'lora_up', 'scale_aware_fusion']),
    dict(type='GradientAndMemoryAuditHook', interval=10),
]
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=100,
    val_interval=100,
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=10),
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=False,
         milestones=[70], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100,
                    max_keep_ckpts=1, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=1),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1,
                       score_thr=0.25, test_out_dir='visualizations'),
)
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)
work_dir = 'experiments/odinw_pothole_high_resolution_scale_lora_r16_diagnostic'
