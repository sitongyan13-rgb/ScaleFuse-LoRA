_base_ = '../baselines/odinw_pothole_fusion_lora_r16.py'

model = dict(
    type='DomainPromptFusionGroundingDINO',
    domain_prompt_count=8,
    domain_prompt_gate_hidden_dim=128,
    domain_prompt_temperature=1.0,
)

custom_hooks = [
    dict(
        type='SetTrainableModulesHook',
        train_only_substrings=[
            'lora_down',
            'lora_up',
            'domain_prompt_fusion',
        ]),
    dict(type='GradientAndMemoryAuditHook', interval=100),
]

work_dir = 'experiments/odinw_pothole_domain_prompt_fusion_lora_r16_seed42'
