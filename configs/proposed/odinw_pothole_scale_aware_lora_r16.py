_base_ = '../baselines/odinw_pothole_fusion_lora_r16.py'

# Pothole's matched LoRA baseline has a much larger validation gap for small
# than large objects (APs 0.378 versus APl 0.666 at seed 42). This module adds
# zero-initialized, image-gated top-down residuals from coarse semantic maps to
# the three finer feature maps. Data, augmentation, optimizer, and evaluator
# remain identical to the rank-16 LoRA baseline.
model = dict(
    type='ScaleAwareFusionLoRAGroundingDINO',
    scale_fusion_num_levels=4,
    scale_fusion_gate_hidden_dim=64,
    scale_fusion_interpolation='bilinear',
)

custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        train_only_substrings=[
            'lora_down',
            'lora_up',
            'scale_aware_fusion',
        ]),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]

work_dir = 'experiments/odinw_pothole_scale_aware_lora_r16_seed42'
