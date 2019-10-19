# EnhGAN
Enhancement GAN

Generate high contrast images for the segmentation task based on trained model on BraTS dataset


# Prerequisites

<p> Linux or OSX </p>
<p> NVIDIA GPU + CUDA CuDNN  </p> 
<p> Keras  </p>
<p> SimpleITK  </p>

# Prepare dataset

1- Put your dataset (here BraTS) on the root address:

2- Create "data_adr.txt" file and determine requirement as bellow:

#############################################
<p>[data] </p>
<p>data_root             = /home/mohammad/input/MICCAI_BraTS17_Data_Training/ </p>
<p>data_names            = config/train_name_all.txt </p>
<p>modality_postfix      = [flair] </p>
<p>file_postfix          = nii.gz </p>
#############################################
<p> Put name of each Subject ID on "train_name_all.txt"  </p> 


# Compute high contrast images based on EnhGAN model

<div class="highlight highlight-source-shell"><pre>
python Enhancement_GAN.py config/data_adr.txt
</pre></div>
