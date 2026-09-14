_base_ = './odinw_pothole_localization_ranking_scale_lora_r16.py'

# Held-out test override. The ensemble parameters are frozen on validation.
test_dataloader = dict(
    dataset=dict(
        ann_file='odinw/pothole/test/annotations_without_background.json',
        data_prefix=dict(img='odinw/pothole/test/'),
    ),
)
test_evaluator = dict(
    ann_file='data/odinw/pothole/test/annotations_without_background.json',
)
