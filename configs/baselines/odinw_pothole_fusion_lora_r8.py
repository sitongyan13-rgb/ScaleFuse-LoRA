_base_ = './generated/pothole_full_finetune.py'

custom_imports = dict(imports=['research.mmdet_plugins'], allow_failed_imports=False)

model = dict(
    type='FusionLoRAGroundingDINO',
    lora_rank=8,
    lora_alpha=16.0,
    lora_dropout=0.05,
    lora_target_roots=(
        'encoder',
        'decoder',
        'bbox_head',
        'memory_trans_fc',
        'text_feat_map',
    ))

custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        train_only_substrings=['lora_down', 'lora_up']),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]

# LoRA conventionally uses a larger learning rate than full-model tuning.
# All other data, schedule, AMP, accumulation, and evaluator settings remain
# identical to the established Pothole baseline.
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(custom_keys={}))

work_dir = 'experiments/odinw_pothole_fusion_lora_r8_seed42'

