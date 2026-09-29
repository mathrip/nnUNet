#!/bin/bash

#! ############################################################
#! sbatch directives 
#! ############################################################
#! Name of the job:
#SBATCH -J run-script
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A CORE-WCHN-MELD-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=35:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Do not change:
#SBATCH -p ampere
#! interactive QoS for debugging only
#!SBATCH --qos=INTR

#! Notes:
#! Charging is determined by GPU number*walltime. 
#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! ############################################################
#! Set up environment 
#! ############################################################

#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
#! added for nnunet
module remove mpi compilers  # remove compiler to not compile python inside
module load python/3.8 


#! ############################################################
#! Code to launch
#! ############################################################

# print some information
JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

# source environment

bash /rds/project/kw350/rds-kw350-meld/meld_fe/code/nnUNet/mr_hpc/launch_training.sh

echo "Process finished"
echo "Time: `date`"
