_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_1024.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(_delete_=True, type='Adam', lr=0.0001, weight_decay=0.0001)