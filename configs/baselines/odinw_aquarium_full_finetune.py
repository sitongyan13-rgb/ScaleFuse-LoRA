_base_ = '../../third_party/mmdetection-3.3.0/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

custom_imports = dict(imports=['research.mmdet_plugins'], allow_failed_imports=False)

# Representative ODinW task. Other tasks use the same hyperparameters and
# replace only data_root/metainfo in their generated task config.
data_root = 'data/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/'
class_name = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
              'stingray')
metainfo = dict(classes=class_name)

load_from = 'weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
model = dict(
    language_model=dict(
        name='weights/bert-base-uncased',
        use_checkpoint=True),
    backbone=dict(with_cp=True),
    encoder=dict(num_cp=6),
    bbox_head=dict(num_classes=len(class_name)))

# Unified baseline transform: fixed 800 x 1333 envelope and horizontal flip.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='train/annotations_without_background.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/annotations_without_background.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=data_root + 'valid/annotations_without_background.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    # Dynamic AMP started at a scale that produced non-finite gradients on
    # the audited Windows/PyTorch stack. Scale 1.0 was verified finite.
    loss_scale=1.0,
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=10))
custom_hooks = [
    dict(type='SetTrainableModulesHook'),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]
randomness = dict(seed=42, deterministic=True)
auto_scale_lr = dict(enable=False, base_batch_size=32)
work_dir = 'experiments/odinw_aquarium_full_finetune'
