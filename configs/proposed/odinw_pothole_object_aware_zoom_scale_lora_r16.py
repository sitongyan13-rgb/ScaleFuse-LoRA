_base_ = './odinw_pothole_scale_aware_lora_r16.py'

# Training-only, inverse-area target sampling followed by a target-containing
# zoom crop. Validation/inference, model, optimizer, prompt, and schedule are
# inherited unchanged from the selected dynamic-scale reference.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='ObjectAwareZoomCrop',
        prob=0.5,
        crop_scale_range=(0.5, 0.8),
        selection_power=0.5,
        min_retained_area=0.5,
    ),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'text', 'custom_entities',
            'zoom_crop_applied', 'zoom_crop_scale', 'zoom_crop_window',
            'zoom_crop_target_original_index', 'zoom_crop_retained_boxes',
            'zoom_crop_target_original_area_fraction',
            'zoom_crop_target_cropped_area_fraction',
            'zoom_crop_target_original_area',
        ),
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = 'experiments/odinw_pothole_object_aware_zoom_scale_lora_r16_seed42'
