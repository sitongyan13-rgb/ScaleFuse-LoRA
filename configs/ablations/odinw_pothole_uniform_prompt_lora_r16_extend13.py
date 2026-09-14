_base_ = './odinw_pothole_uniform_prompt_lora_r16.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=13, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=13,
        by_epoch=True,
        milestones=[11],
        gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best=None,
        max_keep_ckpts=1))

