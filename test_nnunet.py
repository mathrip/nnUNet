'''
1) first activate nnunet env
conda activate /home/mri24/Documents/envs/nnunet

2) then export environment variable:
export nnUNet_raw=/data1/mri24/meldfe_experiments/experiments_monai/dataset_raw
export nnUNet_preprocessed=/data1/mri24/meldfe_experiments/experiments_monai/nnunet_preprocessing
export nnUNet_results=/data1/mri24/meldfe_experiments/experiments_monai/nnunet_results
'''

from nnunetv2.run.run_training import run_training
import torch

dataset_name_or_id = '105'
configuration = '3d_fullres'
fold = 0
tr = 'nnUNetTrainer_20epochs'
p = 'nnUNetPlans'
pretrained_weights = None
num_gpus = 1
npz = False
c = False
val = False
disable_checkpointing = False
val_best = False
device = torch.device('cuda')

run_training(dataset_name_or_id, configuration, fold, tr, p, pretrained_weights,
                 num_gpus, npz, c, val, disable_checkpointing, val_best,
                 device=device)
