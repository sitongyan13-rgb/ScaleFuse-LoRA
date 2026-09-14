_base_ = '../baselines/odinw_pothole_fusion_lora_r16.py'

# Controlled convergence sensitivity: resume epoch 12 and continue at the
# already-decayed learning rate through epoch 20.
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

# Do not let restored best-checkpoint metadata mutate the original work dir.
# All eight new epoch checkpoints are retained for offline 1--20 selection.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best=None,
        max_keep_ckpts=8))

work_dir = 'experiments/odinw_pothole_lora_r16_seed42_extend20'
