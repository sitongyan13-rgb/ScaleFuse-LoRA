_base_ = './pothole_frozen_text.py'

data_root = 'data/'
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='odinw/pothole/test/annotations_without_background.json',
        data_prefix=dict(img='odinw/pothole/test/')))
test_evaluator = dict(
    ann_file=data_root +
    'odinw/pothole/test/annotations_without_background.json')
work_dir = 'experiments/odinw_pothole_frozen_text_test'
