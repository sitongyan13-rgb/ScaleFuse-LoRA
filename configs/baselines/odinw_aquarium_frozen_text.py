_base_ = './odinw_aquarium_full_finetune.py'

custom_imports = dict(imports=['research.mmdet_plugins'], allow_failed_imports=False)
custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        freeze_prefixes=['language_model']),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.0),
        }))
work_dir = 'experiments/odinw_aquarium_frozen_text'
