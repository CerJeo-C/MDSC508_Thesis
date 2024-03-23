#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --mem=20GB


######## Environment Variables #########
export PATH=/home/sergiu.cociuba/miniconda3/bin:$PATH

####### Script Below ############
source activate blptl                                                             # Activates the monai environment

script_dir="/home/sergiu.cociuba/BoneLab/508/"                                    # Specifies where the python scripts are
data_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/output_train_validate.csv"  # Where the test directory is
test_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/output_test.csv"
file="linear_regression.py"                                                         # Which python script you want to run
bone="t"

python ${script_dir}${file} ${data_dir} ${test_dir} ${bone} ${SLURM_JOB_ID}
echo "JobID: " ${SLURM_JOB_ID}


