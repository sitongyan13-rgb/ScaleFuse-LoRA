_base_ = './generated/aquarium_full_finetune.py'

custom_imports = dict(imports=['research.mmdet_plugins'],
                      allow_failed_imports=False)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False)
test_dataloader = val_dataloader

# One real ODinW optimizer step only. Full benchmark settings remain in the
# inherited generated config and are not altered by this smoke overlay.
optim_wrapper = dict(accumulative_counts=1)
custom_hooks = [
    dict(type='GradientAndMemoryAuditHook', interval=1)
]
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=1,
    val_interval=1)
param_scheduler = []
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1,
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=1))
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)
randomness = dict(seed=42, deterministic=True)
work_dir = 'experiments/smoke_odinw_aquarium_1iter'
