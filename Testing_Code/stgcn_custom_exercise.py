_base_ = '../mmaction2/configs/_base_/default_runtime.py'

loso_dir ='../loso'
ann_file = '../datasets/8class-upperbodyonly/loso_split_s01.pkl'

load_from = 'modified_checkpoint.pth' # See jupyternotebook for code to generate this

# Model settings
custom_layout = dict(
    num_node= 22,
    inward= [
        (1, 0),  # Waist -> Spine
        (2, 1),  # Spine -> Chest
        (3, 2),  # Chest -> Neck
        (4, 3),  # Neck -> Head
        (5, 4),  # Head -> Head tip
        (6, 3),  # Neck -> Left collar
        (7, 6),  # Left collar -> Left upper arm
        (8, 7),  # Left upper arm -> Left forearm
        (9, 8),  # Left forearm -> Left hand
        (10, 3), # Neck -> Right collar
        (11, 10), # Right collar -> Right upper arm
        (12, 11), # Right upper arm -> Right forearm
        (13, 12), # Right forearm -> Right hand
        (14, 0),  # Waist -> Left upper leg
        (15, 14), # Left upper leg -> Left lower leg
        (16, 15), # Left lower leg -> Left foot
        (17, 16), # Left foot -> Left leg toes
        (18, 0),  # Waist -> Right upper leg
        (19, 18), # Right upper leg -> Right lower leg
        (20, 19), # Right lower leg -> Right foot
        (21, 20)  # Right foot -> Right leg toes
    ],
    center= 0
)
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        graph_cfg=dict(layout=custom_layout, mode="stgcn_spatial"),
        in_channels=3  # 3D input (xyz)
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=8,  # Set this to the number of action classes
        in_channels=256
    )
)

# Dataset settings
dataset_type = 'PoseDataset'

train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    # dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    # dict(
    #     type='UniformSampleFrames', clip_len=100, num_clips=10,
    #     test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train'))
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=16, val_begin=1, val_interval=1
)
val_cfg = dict(type='ValLoop')
val_evaluator = [
    dict(type='AccMetric')
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),  # Saves the best checkpoint
    # early_stop=dict(
    #     type='EarlyStoppingHook', 
    #     patience=5,  # Number of epochs with no improvement before stopping
    #     monitor='acc/top1',  # Metric to monitor (ensure it's available in your logs)
    #     rule='greater'  # 'greater' for metrics that should increase (like accuracy)
    # )
)

# Optimizer and scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=1e-6, momentum=0.9, weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2))


