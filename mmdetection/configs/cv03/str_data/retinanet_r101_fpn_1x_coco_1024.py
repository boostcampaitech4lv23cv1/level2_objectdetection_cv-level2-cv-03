_base_ = '../../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

optimizer = dict(_delete_=True, type='Adam', lr=0.0001, weight_decay=0.0001)