_base_ = './pothole_full_finetune.py'

data_root = 'data/'
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='odinw/pothole/test/annotations_without_background.json',
        data_prefix=dict(img='odinw/pothole/test/')))
test_evaluator = dict(
    ann_file=data_root +
    'odinw/pothole/test/annotations_without_background.json')
work_dir = 'experiments/odinw_pothole_full_finetune_test'
