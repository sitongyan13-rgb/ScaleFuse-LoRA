_base_ = './odinw_pothole_scale_aware_lora_r16.py'

# Validation error analysis shows that dynamic scale fusion raises small-object
# AR from 0.545 to 0.595 and mean small-GT best IoU by 0.030, while APs rises
# only 0.0026. This no-parameter head aligns matched positive token targets to
# detached box IoU so classification confidence reflects localization quality.
model = dict(
    bbox_head=dict(
        type='QualityAlignedGroundingDINOHead',
        quality_target_power=1.0,
    ),
)

work_dir = 'experiments/odinw_pothole_quality_aligned_scale_lora_r16_seed42'
