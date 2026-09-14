_base_ = './odinw_pothole_scale_aware_lora_r16.py'

# Held-out test override. This config is evaluation-only and must not be used
# to choose checkpoints, thresholds, prompts, or ensemble constituents.
test_dataloader = dict(
    dataset=dict(
        ann_file='odinw/pothole/test/annotations_without_background.json',
        data_prefix=dict(img='odinw/pothole/test/'),
    ),
)
test_evaluator = dict(
    ann_file='data/odinw/pothole/test/annotations_without_background.json',
)
