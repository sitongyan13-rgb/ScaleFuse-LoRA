_base_ = '../proposed/odinw_pothole_domain_prompt_fusion_lora_r16.py'

# The inserted LoRA branches remain exact-zero and frozen. Only the learned
# visual-conditioned prompt bank/gate/projection is optimized.
custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        train_only_substrings=['domain_prompt_fusion'],
    ),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]

work_dir = 'experiments/odinw_pothole_prompt_only_seed42'
