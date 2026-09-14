_base_ = './odinw_pothole_scale_aware_lora_r16.py'

# Matching-only, scale-adaptive L1/GIoU weighting. Denoising, classification,
# data, prompts, optimizer, schedule, and inference remain unchanged.
model = dict(
    bbox_head=dict(
        type='ScaleAdaptiveRegressionGroundingDINOHead',
        regression_reference_area=0.01,
        regression_max_weight=2.0,
    ),
)

work_dir = (
    'experiments/'
    'odinw_pothole_scale_adaptive_regression_scale_lora_r16_seed42'
)
