_base_ = './odinw_pothole_fusion_lora_r8.py'

model = dict(
    lora_rank=16,
    lora_alpha=32.0,
)

work_dir = 'experiments/odinw_pothole_fusion_lora_r16_seed42'
