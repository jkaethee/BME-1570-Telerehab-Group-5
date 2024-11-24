# BME-1570-Telerehab-Group-5

## Instructions for running:
1. Clone the repo
2. Run all cells in the UI-PRMD20pre-processing.ipynb notebook to generate the preprocessed `.npy` files of 3D skeletal data
3. Navigate to the `datasets` folder and run `2class-all/generate_splits.ipynb` and `80-20-split/generate_splits/ipynb`. This will generate the necessary PKL files for hyperparameter tuning and LOSO validation.
4. Open Anaconda Prompt and run the following commands to set up an environment for this project:
   ```
    conda create --name mmactionenv python=3.8 -y
    conda activate openmmlab
    conda install pytorch torchvision -c pytorch
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.1.0
    mim install mmdet==3.2.0
    mim install mmpose
    pip install mmaction2
   ```
5. Navigate to the `mmaction2` directory and replace the tools/train.py file with the `train.py` file provided in the `Testing_Code` directory. This file has modified the code to run hyperparameter tuning and LOSO validation.
6. If you are doing hyperparameter tuning please uncomment the call for `hyperparam_grid_search(cfg)` and comment out the code below it. If you are doing LOSO validation please leave the file as is.
7. Whether you are doing hyperparameter tuning or LOSO validation, the command to train and validate the model is the same:
   `python ../mmaction2/tools/train.py stgcn_custom_exercise.py --seed 0 --deterministic`
8. You will find the results from the process under its respective folders in the `Testing_Code` directory. To examine the results per split for LOSO, please navigate to `Testing_Code\loso_2class_all\loso_split_s10.pkl_lr_0.001_bs_16_repeat_3\loso_results.txt` where the metrics will be listed for each split. 
