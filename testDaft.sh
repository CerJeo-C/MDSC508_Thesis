#!/bin/bash
####### Reserve computing resources #############   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu-a100 --gres=gpu:1                                           # Maximum allocation for gpu as of now
#SBATCH --time=24:00:00
#SBATCH --mem=256GB

######## Environment Variables #########
export PATH=/home/sergiu.cociuba/miniconda3/bin:$PATH

####### Script Below ############
. /home/sergiu.cociuba/conda_init/sergiu.sh
conda activate medcam                                                             # Activates the monai environment
cd "/home/sergiu.cociuba/BoneLab/DAFT/daft/"
#script_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/"                     # Specifies where the python scripts are
data_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/Data03/Tibia/Test"                        # Where the test directory is
file="DaftTest.py"                                                             # Which python script you want to run
bone="t"
model="/home/sergiu.cociuba/BoneLab/DAFT/daft/Results/NewResults/Radius/Daft_train/Best_Model_Fold_4_Epoch_9.pth"

python ${file} ${data_dir} ${bone} ${model}
echo "JobID: " ${SLURM_JOB_ID}


