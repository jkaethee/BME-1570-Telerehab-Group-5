# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set random seeds
    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


import matplotlib.pyplot as plt
from mmengine.hooks import Hook
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import itertools

class FrameDirLoggerHook(Hook):
    connections = [
        (1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
        (6, 3), (7, 6), (8, 7), (9, 8), (10, 3),
        (11, 10), (12, 11), (13, 12), (14, 0), (15, 14),
        (16, 15), (17, 16), (18, 0), (19, 18), (20, 19),
        (21, 20)
    ]

    def visualize_skeleton_animation(self, skeleton_data, title="Skeleton Animation", save_path="skeleton_animation.gif"):
        """
        Visualize 3D skeleton data as an animation and save it as a GIF.

        :param skeleton_data: A numpy array of shape (num_frames, num_joints, 3)
        :param title: Title of the animation
        :param save_path: Path to save the GIF
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Debug: Print data stats
        print(f"Skeleton data shape: {skeleton_data.shape}")
        print(f"First frame data:\n{skeleton_data[0]}")

        # Scale the data for better visualization
        skeleton_data -= skeleton_data.mean(axis=(0, 1))  # Center the data
        skeleton_data /= skeleton_data.std(axis=(0, 1))   # Normalize to unit variance

        # Set plot limits (adjust based on your dataset)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(title)

        scatter = ax.scatter([], [], [], c='r', label='Joints')
        lines = [ax.plot([], [], [], 'b-')[0] for _ in self.connections]

        def init():
            """Initialize the animation."""
            scatter._offsets3d = ([], [], [])
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scatter] + lines

        def update(frame):
            """Update the animation for each frame."""
            frame_data = skeleton_data[frame]  # Extract data for the current frame

            # Update joint positions
            scatter._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])

            # Update bone connections
            for line, (start, end) in zip(lines, self.connections):
                line.set_data(
                    [frame_data[start, 0], frame_data[end, 0]],
                    [frame_data[start, 1], frame_data[end, 1]]
                )
                line.set_3d_properties(
                    [frame_data[start, 2], frame_data[end, 2]]
                )

            return [scatter] + lines

        # Create the animation
        ani = FuncAnimation(fig, update, frames=skeleton_data.shape[0], init_func=init, blit=False)

        # Save the animation as a GIF
        ani.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved as {save_path}")
        plt.close(fig)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """
        Log and visualize skeleton data as animation after each training iteration.
        """
        # inputs = data_batch['inputs']  # Assuming this contains the 3D skeleton data

        # for sample_idx, sample in enumerate(inputs):
        #     # Extract skeleton data for all frames of the first person
        #     skeleton_data = sample[0, 0].cpu().numpy()  # Shape: [num_frames, num_joints, 3]

        #     print("Skeleton data: ", skeleton_data.shape)

        #     # Visualize as animation and save as GIF
        #     self.visualize_skeleton_animation(skeleton_data, title=f"Skeleton Animation - Batch {batch_idx}, Sample {sample_idx}", save_path=f"skeleton_animation_batch_{batch_idx}_sample_{sample_idx}.gif")
        #     break  # Visualize only one sample per iteration

def hyperparam_grid_search(cfg):
    # Define hyperparameter search space
    param_grid = {
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [16, 32],
        "repeat_dataset": [1, 2, 3]
    }

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        param_grid["learning_rate"],
        param_grid["batch_size"],
        param_grid["repeat_dataset"]
    ))

    best_result = 0
    best_params = None

    for learning_rate, batch_size, repeat_dataset in param_combinations:
        # Modify parameters in the config
        cfg.optim_wrapper.optimizer.lr = learning_rate
        cfg.train_dataloader.batch_size = batch_size
        cfg.val_dataloader.batch_size = batch_size
        cfg.train_dataloader.dataset.times = repeat_dataset
        
        # Set unique work directory
        cfg.work_dir = f'./hyperparam_2class_all/lr_{learning_rate}_bs_{batch_size}_repeat_{repeat_dataset}'

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # Check dataset
        assert len(runner.train_dataloader.dataset) > 0, "Training dataset is empty!"
        assert len(runner.val_dataloader.dataset) > 0, "Validation dataset is empty!"

        runner.register_hook(FrameDirLoggerHook())

        # start training
        runner.train()

        # Evaluate the model and fetch the desired metric (e.g., accuracy)
        eval_metrics = runner.val()
        current_metric = eval_metrics.get("acc/top1")

        print(f"Final val for this run (LR, batch_size, repeat_dataset): {(learning_rate, batch_size, repeat_dataset)}, Results: {eval_metrics}")

        # Update the best result if the current metric is better
        if current_metric > best_result:
            best_result = current_metric
            best_params = (learning_rate, batch_size, repeat_dataset)

    # Print the best combination of hyperparameters and the result
    print(f"Best Parameters (LR, batch_size, repeat_dataset): {best_params}, Best Result (acc/top1): {best_result}")


def loso_validation(cfg, ann_file, file_name):
    hyperparams = [1e-3, 16, 3] # Chosen hyperparameters
    # Modify parameters in the config
    cfg.optim_wrapper.optimizer.lr = hyperparams[0]
    cfg.train_dataloader.batch_size = hyperparams[1]
    cfg.val_dataloader.batch_size = hyperparams[1]
    cfg.train_dataloader.dataset.times = hyperparams[2]
    cfg.train_dataloader.dataset.dataset.ann_file = ann_file
    cfg.val_dataloader.dataset.ann_file = ann_file
    
    # Set unique work directory
    dir_path = f'./loso_2class_s07/{file_name}_lr_{hyperparams[0]}_bs_{hyperparams[1]}_repeat_{hyperparams[2]}'
    cfg.work_dir = dir_path

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # Check dataset
    assert len(runner.train_dataloader.dataset) > 0, "Training dataset is empty!"
    assert len(runner.val_dataloader.dataset) > 0, "Validation dataset is empty!"

    runner.register_hook(FrameDirLoggerHook())

    # start training
    runner.train()
    eval_metrics = runner.val()
    print(eval_metrics)
    return eval_metrics, dir_path

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # Uncomment line below to run grid search    
    # hyperparam_grid_search(cfg)

    from pathlib import Path
    import torch

    # Reproducability
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    data_dir = Path("../datasets/2class-all")
    eval_metrics_list = []
    for pkl_file in data_dir.iterdir():
        if (pkl_file.is_file and pkl_file.suffix == '.pkl'):
            ann_file = str(pkl_file)
            eval_metrics, dir_path = loso_validation(cfg, ann_file, pkl_file.name)
            eval_metrics_list.append({"pkl_file": pkl_file.name, "metrics": eval_metrics})
    
            with open(f"{dir_path}/loso_results.txt", "w") as f:
                for item in eval_metrics_list:
                    f.write(f"{item}\n")

if __name__ == '__main__':
    main()
