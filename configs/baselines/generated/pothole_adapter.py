# Generated mechanically by scripts/generate_odinw_configs.py.
# Dataset paths/classes come from the pinned official GLIP ODinW-13 YAML.
_base_ = '../odinw_aquarium_adapter.py'

task_name = 'pothole'
class_name = ('pothole',)
metainfo = dict(classes=class_name)
data_root = 'data/'

model = dict(bbox_head=dict(num_classes=len(class_name)))
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='odinw/pothole/train/annotations_without_background.json',
        data_prefix=dict(img='odinw/pothole/train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='odinw/pothole/valid/annotations_without_background.json',
        data_prefix=dict(img='odinw/pothole/valid/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'odinw/pothole/valid/annotations_without_background.json')
test_evaluator = val_evaluator
work_dir = 'experiments/odinw/pothole/adapter'
