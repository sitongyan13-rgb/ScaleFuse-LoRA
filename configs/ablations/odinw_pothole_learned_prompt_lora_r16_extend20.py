_base_ = '../proposed/odinw_pothole_domain_prompt_fusion_lora_r16.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
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
        end=20,
        by_epoch=True,
        milestones=[11],
        gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best=None,
        max_keep_ckpts=8))

work_dir = 'experiments/odinw_pothole_learned_prompt_lora_r16_seed42_extend20'
