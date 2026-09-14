_base_ = './odinw_aquarium_full_finetune.py'

custom_imports = dict(imports=['research.mmdet_plugins'], allow_failed_imports=False)
model = dict(type='AdapterGroundingDINO', adapter_bottleneck=16)
custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        train_only_prefixes=['vision_adapter', 'text_adapter']),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(custom_keys={}))
work_dir = 'experiments/odinw_aquarium_adapter'
