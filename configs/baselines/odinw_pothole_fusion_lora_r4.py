_base_ = './odinw_pothole_fusion_lora_r8.py'

model = dict(
    lora_rank=4,
    lora_alpha=8.0,
)

work_dir = 'experiments/odinw_pothole_fusion_lora_r4_seed42'
