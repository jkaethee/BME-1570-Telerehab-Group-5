_base_ = '../mmaction2/configs/_base_/default_runtime.py'

loso_dir ='../loso'
ann_file = '../loso_split_s01.pkl'
# Model settings
custom_layout = dict(
    num_node= 22,  # Total number of nodes in your skeleton
    inward= [
        (1, 0), (2, 1), (3, 2),          # Spine: Waist -> Spine -> Chest -> Neck
        (4, 3), (5, 4),                  # Neck -> Head -> Head tip
        (6, 3), (8, 6), (9, 8),         # Left side: Neck -> Left collar -> Left upper arm -> Left forearm -> Left hand
        (16, 9), (17, 16),              # Left hand -> Left lower leg -> Left leg toes
        (10, 3), (11, 10), (13, 11),     # Right side: Neck -> Right collar -> Right upper arm -> Right forearm -> Right hand
        (20, 13), (21, 20),              # Right hand -> Right foot -> Right leg toes
        (14, 2), (15, 14),               # Left upper leg -> Left lower leg
        (18, 2), (19, 18)                # Right upper leg -> Right lower leg
    ],
    center= 0  # Center node is the spine
)
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        graph_cfg=dict(layout=custom_layout, mode="stgcn_spatial"),
        in_channels=3,  # 3D input (xyz)
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=2,  # Set this to the number of action classes
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')
    ),
    train_cfg=dict(),
    test_cfg=dict()
)

# Dataset settings
dataset_type = 'PoseDataset'
ann_file_train = 'data/custom/train.pkl'
ann_file_val = 'data/custom/val.pkl'

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='xsub_train'))

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
val_evaluator = [dict(type='AccMetric')]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),  # Saves the best checkpoint
    early_stop=dict(
        type='EarlyStoppingHook', 
        patience=5,  # Number of epochs with no improvement before stopping
        monitor='acc/top1',  # Metric to monitor (ensure it's available in your logs)
        rule='greater'  # 'greater' for metrics that should increase (like accuracy)
    )
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
optimizer=dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True))
