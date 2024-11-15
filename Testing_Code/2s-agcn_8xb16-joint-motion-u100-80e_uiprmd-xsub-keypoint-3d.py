_base_ = '2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py'  # Base config

# Define the dataset type (UI-PRMD dataset)
dataset_type = 'PoseDataset'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]


# Function to create the dataset splits (train and test for each fold)
def get_train_val_split(test_subject, pickle_dir):
    all_subjects = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
    train_data = [os.path.join(pickle_dir, subject) for subject in all_subjects if subject != test_subject]
    val_data = os.path.join(pickle_dir, test_subject)
    return train_data, val_data


train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=[],  # You will provide the specific train data path in the script
            pipeline=train_pipeline,
            split=None,  # No split, as we're manually handling training samples
        )
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=[],  # You will provide the specific val data path in the script
        pipeline=val_pipeline,
        split=None,  # No split, as we're manually handling the split
        test_mode=True
    )
)

# Adding checkpoint configuration to save model periodically
checkpoint_config = dict(
    interval=5,  # Save checkpoint every 5 epochs
    save_optimizer=True,
    save_by_epoch=True,
    out_dir=work_dir
)

# Adding evaluation configuration to evaluate periodically
evaluation = dict(
    interval=2,  # Evaluate the model every 5 epochs
    metric='accuracy'
)

work_dir = '/Users/jennywei/mmaction2' # Set the directory where you want to save logs and models

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', log_file=f'{work_dir}/training_log.txt'),  # Save logs to a text file
        dict(type='TensorboardLoggerHook', log_dir=f'{work_dir}/tensorboard')  # Save to TensorBoard
    ]
)
