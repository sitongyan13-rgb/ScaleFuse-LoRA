_base_ = '../proposed/odinw_pothole_domain_prompt_fusion_lora_r16.py'

# Static equal weighting isolates whether image-conditioned gate weights add
# value beyond a learned continuous prompt residual and rank-16 Fusion-LoRA.
model = dict(domain_prompt_gate_mode='uniform')

work_dir = 'experiments/odinw_pothole_uniform_prompt_lora_r16_seed42'
