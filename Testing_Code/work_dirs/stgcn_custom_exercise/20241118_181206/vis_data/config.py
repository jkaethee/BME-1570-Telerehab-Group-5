ann_file = '../datasets/12class-lowerbodyonly/loso_split_s01'
custom_layout = dict(
    center=0,
    inward=[
        (
            1,
            0,
        ),
        (
            2,
            1,
        ),
        (
            3,
            2,
        ),
        (
            4,
            3,
        ),
        (
            5,
            4,
        ),
        (
            6,
            3,
        ),
        (
            7,
            6,
        ),
        (
            8,
            7,
        ),
        (
            9,
            8,
        ),
        (
            10,
            3,
        ),
        (
            11,
            10,
        ),
        (
            12,
            11,
        ),
        (
            13,
            12,
        ),
        (
            14,
            0,
        ),
        (
            15,
            14,
        ),
        (
            16,
            15,
        ),
        (
            17,
            16,
        ),
        (
            18,
            0,
        ),
        (
            19,
            18,
        ),
        (
            20,
            19,
        ),
        (
            21,
            20,
        ),
    ],
    num_node=22)
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'modified_checkpoint.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
loso_dir = '../loso'
model = dict(
    backbone=dict(
        gcn_adaptive='init',
        graph_cfg=dict(
            layout=dict(
                center=0,
                inward=[
                    (
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                    ),
                    (
                        3,
                        2,
                    ),
                    (
                        4,
                        3,
                    ),
                    (
                        5,
                        4,
                    ),
                    (
                        6,
                        3,
                    ),
                    (
                        7,
                        6,
                    ),
                    (
                        8,
                        7,
                    ),
                    (
                        9,
                        8,
                    ),
                    (
                        10,
                        3,
                    ),
                    (
                        11,
                        10,
                    ),
                    (
                        12,
                        11,
                    ),
                    (
                        13,
                        12,
                    ),
                    (
                        14,
                        0,
                    ),
                    (
                        15,
                        14,
                    ),
                    (
                        16,
                        15,
                    ),
                    (
                        17,
                        16,
                    ),
                    (
                        18,
                        0,
                    ),
                    (
                        19,
                        18,
                    ),
                    (
                        20,
                        19,
                    ),
                    (
                        21,
                        20,
                    ),
                ],
                num_node=22),
            mode='stgcn_spatial'),
        in_channels=3,
        type='STGCN'),
    cls_head=dict(in_channels=256, num_classes=12, type='GCNHead'),
    type='RecognizerGCN')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.0005,
        momentum=0.9,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005))
param_scheduler = [
    dict(
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_pipeline = [
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=30, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        dataset=dict(
            ann_file='../datasets/12class-lowerbodyonly/loso_split_s01',
            pipeline=[
                dict(type='PoseDecode'),
                dict(num_person=1, type='FormatGCNInput'),
                dict(type='PackActionInputs'),
            ],
            split='xsub_train',
            type='PoseDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='../datasets/12class-lowerbodyonly/loso_split_s01',
        pipeline=[
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\stgcn_custom_exercise'
