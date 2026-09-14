_base_ = './odinw_pothole_scale_aware_lora_r16.py'

# Preserve upstream binary token supervision and add a final-layer auxiliary
# loss that ranks well-localized matches above weaker matches/hard negatives.
# The inference graph and parameter count are unchanged.
model = dict(
    bbox_head=dict(
        type='LocalizationRankingGroundingDINOHead',
        ranking_loss_weight=0.1,
        ranking_min_quality_gap=0.05,
        ranking_temperature=0.5,
        ranking_hard_negative_k=20,
        ranking_hard_negative_weight=0.25,
    ),
)

work_dir = 'experiments/odinw_pothole_localization_ranking_scale_lora_r16_seed42'
