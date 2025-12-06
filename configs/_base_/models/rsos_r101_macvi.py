# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',  
    backbone=dict(
        type='ResNetV1c',
        depth=101,  
        num_stages=4,
        out_indices=(0, 1, 2, 3), 
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='RSOSHead_MaCVi',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2,3],
        fppm=dict(c1=2048, c2=256),
        ffm=dict(in_channels=1024, out_channels=512, scale_factor=4), 
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=2.0),
            dict(type='FocalLoss', loss_name='loss_focal', loss_weight=1.0)
        ]),  
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            in_index=1,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=3,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            in_index=0,
            channels=128,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=3,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),   
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))