_base_ = '../../third_party/mmdetection-3.3.0/configs/grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py'

# The original checkpoint was converted by the official MMDetection script.
load_from = 'weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
model = dict(language_model=dict(name='weights/bert-base-uncased'))

randomness = dict(seed=42, deterministic=True)
work_dir = 'experiments/odinw13_zero_shot'

