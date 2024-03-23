#!/bin/bash
####### Reserve computing resources #############   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --mem=32GB


######## Environment Variables #########
export PATH=/home/sergiu.cociuba/miniconda3/bin:$PATH

####### Script Below ############
source activate blptl                                                                                                   

######################################## TIBA #################################################################################################
script_dir="/home/sergiu.cociuba/BoneLab/508/"                                                          # Specifies where the python scripts are
file="Preprocessing_main.py"                                                                            # Which python script you want to run
image_dir="/work/boyd_lab/data/VITD/Tibia"                                                              # Directory where your images are stored
pattern="*.AIM"                                                                                         # File extension your image data has
tabular_data="/home/sergiu.cociuba/BoneLab/DAFT/daft/output_train_validate.csv"
output_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/Data03/Tibia/Train"                                            # Output Directory

python ${script_dir}${file} ${image_dir} ${pattern} ${tabular_data} ${output_dir}                         # Execute the python file specified

######################################## RADIUS #################################################################################################
script_dir="/home/sergiu.cociuba/BoneLab/508/"                                                          # Specifies where the python scripts are
file="Preprocessing_main.py"                                                                            # Which python script you want to run
image_dir="/work/boyd_lab/data/VITD/Radius"                                                              # Directory where your images are stored
pattern="*.AIM"                                                                                         # File extension your image data has
tabular_data="/home/sergiu.cociuba/BoneLab/DAFT/daft/output_train_validate.csv"
output_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/Data03/Radius/Train"                                            # Output Directory

python ${script_dir}${file} ${image_dir} ${pattern} ${tabular_data} ${output_dir}                         # Execute the python file specified

######################################## TIBA TEST #################################################################################################
script_dir="/home/sergiu.cociuba/BoneLab/508/"                                                          # Specifies where the python scripts are
file="Preprocessing_main.py"                                                                            # Which python script you want to run
image_dir="/work/boyd_lab/data/VITD/Tibia"                                                              # Directory where your images are stored
pattern="*.AIM"                                                                                         # File extension your image data has
tabular_data="/home/sergiu.cociuba/BoneLab/DAFT/daft/output_test.csv"
output_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/Data03/Tibia/Test"                                            # Output Directory

python ${script_dir}${file} ${image_dir} ${pattern} ${tabular_data} ${output_dir}                         # Execute the python file specified

######################################## RADIUS TEST #################################################################################################
script_dir="/home/sergiu.cociuba/BoneLab/508/"                                                          # Specifies where the python scripts are
file="Preprocessing_main.py"                                                                            # Which python script you want to run
image_dir="/work/boyd_lab/data/VITD/Radius"                                                              # Directory where your images are stored
pattern="*.AIM"                                                                                         # File extension your image data has
tabular_data="/home/sergiu.cociuba/BoneLab/DAFT/daft/output_test.csv"
output_dir="/home/sergiu.cociuba/BoneLab/DAFT/daft/Data03/Radius/Test"                                            # Output Directory

python ${script_dir}${file} ${image_dir} ${pattern} ${tabular_data} ${output_dir}                         # Execute the python file specified

conda deactivate
echo "JobID: " ${SLURM_JOB_ID}


