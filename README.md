# MDSC508_Thesis

This project aimed to use HR-pQCT images and tabular data to train 5 different models, a linear regression, support vector regresison, a random forest, a ResNet, and a ResNet modified with a dynamic affine feature map transform (DAFT). This project was performed under the supervison of Dr. Steven Boyd at the University of Calgary in 2024.

Documentation for the DAFT model:
https://github.com/ai-med/DAFT

An environment containing the required libraries to open HR-pQCT images was created by Dr. Steven Boyd's lab can be installed here:
https://github.com/Bonelab/Bonelab

preprocessingTabularData.py is used to preprocess the tabular data
PreprocessingMain.py contains the code to preprocess and pickle the HR-pQCT images with their labels and tabular data
preprocessScript508.sh is the script used to run the preprocessing scripts on ARC, which uses SLURM

The Models folder provides all the scripts with each of the 5 models. it also contains the scripts used to run them on ARC, and to run the test Dataset on them.

The MedCamCode folder provides the code to save HR-pQCT images as Nii files, upscale the attention maps, and generate a heatmap on the attention map. 

Install ITK snap, load the original image, then load the modified attention map from the ColorMap and CreateAttentionMap scripts.
Treat the attention map as a segementation, and a heatmap will be overlayed over the original image. Adjust the pixel colors as desired.
