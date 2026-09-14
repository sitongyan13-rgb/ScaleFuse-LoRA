_base_ = './odinw_pothole_scale_aware_lora_r16.py'

# Fixed two-view test-time augmentation: original + horizontal flip at the
# baseline 800x1333 resize. Predictions are mapped back and merged by NMS.
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RandomFlip', prob=0.0),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=(
                        'img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction', 'text',
                        'custom_entities',
                    ),
                ),
            ],
        ],
    ),
]

work_dir = 'experiments/odinw_pothole_scale_aware_lora_r16_flip_tta_validation'
