#!/bin/bash

#! ############################################################
#! sbatch directives 
#! ############################################################
#! Name of the job:
#SBATCH -J run_training
#! Which project should be charged:
#SBATCH -A CORE-WCHN-MELD-SL2-CPU
#! Name of the partition:
#SBATCH -p icelake
#! How many whole nodes and tasks should be allocated?
#SBATCH --nodes=1
#SBATCH --ntasks=8
#! How much wallclock time will be required?
#SBATCH --time=3:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
# ! Array to launch
#! SBATCH --array=1-54

#! Notes:
#! Charging is determined by cpu number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*76 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3380 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

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
module load rhel8/default-icl              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
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

env=/rds/project/kw350/rds-kw350-meld/envs/nnunet

# run file 
bash /rds/project/kw350/rds-kw350-meld/meld_fe/code/nnUNet/mr_hpc/launch_training.sh