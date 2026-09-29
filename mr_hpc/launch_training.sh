#!/bin/bash

base_path=/rds/project/kw350/rds-kw350-meld/meld_fe/experiments_meld_find

# initialise the dataset and paths
input_folder=${base_path}/dataset_raw
preprocessed_folder=${base_path}/dataset_nnunet_preprocessing
output_folder_base=${base_path}/experiments
pretrained_model_path=${base_path}/nnunet_pretrained_models

# generate result folder ID
rand_id=$(tr -dc 'A-Z' < /dev/urandom | head -c 3)
folder_name="$(date +%y-%m-%d)_nnunet_results_${rand_id}"
output_folder=${output_folder_base}/${folder_name}

dataset_id=005
configuration=3d_fullres
folds=0

# initialise the nnunet paths
export nnUNet_raw=$input_folder
export nnUNet_preprocessed=$preprocessed_folder
export nnUNet_results=$output_folder
export nnssl_pretrained_models=$pretrained_model_path
export nnUNet_n_proc_DA=32

# env
env=/rds/project/kw350/rds-kw350-meld/envs/nnunet_pretrain


# 1) Standard planning to create a base plan (nnUNetPlans.json).
#    Lightweight (plan only, no default preprocessing):
# $env/bin/nnUNetv2_extract_fingerprint -d $dataset_id
# $env/bin/nnUNetv2_plan_experiment -d $dataset_id
#    (nnUNetv2_plan_and_preprocess -d $dataset_id also works but additionally
#     preprocesses the default plan, which is not needed here.)

# 2) Create the pretraining plan and preprocess the dataset to match the pre-training.
# $env/bin/nnUNetv2_preprocess_like_nnssl \
#     -d $dataset_id \
#     -n nnssl_pretrained \
#     -pc https://huggingface.co/AnonRes/ResEncL-OpenMind-MAE \
#     -am like_pretrained

# extract sampling location
# $env/bin/nnUNetv2_extract_sampling_locations -d 005 -plans_name ptPlans__nnssl_pretrained____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z

# This writes a plans file named  ptPlans__<UniquePretrainingName>...
# It contains everything needed (weights path, key mappings, architecture),
# so you no longer need to pass the checkpoint at training time.

# # # 3) Finetune. Note: dataset / configuration / fold are POSITIONAL arguments.
$env/bin/nnUNetv2_train_pretrained $dataset_id $configuration $folds \
    -p ptPlans__nnssl_pretrained____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z \
    -tr PretrainedTrainer