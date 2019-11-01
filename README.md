# EnhGAN
Enhancement GAN

We present a novel architecture based on conditional generative adversarial networks (cGANs) to improve the lesion contrast for segmentation.

You can find detailed results (Team name: Hamghalam) on BraTS 2019 dataset on:
<p> - Validation Phase Leaderboard 2019 - </p>
<p> https://www.cbica.upenn.edu/BraTS19/lboardValidation.html </p>
<p>  - Training Phase Leaderboard 2019 - </p>
<p> https://www.cbica.upenn.edu/BraTS19/lboardTraining.html </p>


![](https://github.com/hamghalam/EnhGAN/blob/master/high_contrast.PNG)

# High Tissue Contrast MRI Synthesis

<!--   <a href="https://youtu.be/JglyZNLu3ug"><img src="https://github.com/hamghalam/EnhGAN/blob/master/youtub.PNG" 
<!-- alt="https://youtu.be/JglyZNLu3ug"></a> -->
![Alt Text](https://github.com/hamghalam/EnhGAN/blob/master/combined2.gif)


# Prerequisites

<p> A CUDA compatable GPU with memory not less than 12GB is recommended for training. For testing only, a smaller GPU should be suitable. </p>
<p> Linux or OSX </p>
<p> NVIDIA GPU + CUDA CuDNN  </p> 
<p> Keras  </p>
<p> SimpleITK  </p>

# Pretrained model

Download pretrained model (trained on BraTS dataset.) on this address:

<p> https://drive.google.com/open?id=1Gc-gbrq-KoI67tgn-nFiCSdY5jd0dO4y </p>


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


# To compute high contrast images based on EnhGAN model, run this command on Linux terminal:

<div class="highlight highlight-source-shell"><pre>
python Enhancement_GAN.py config/data_adr.txt
</pre></div>

# How to download data
BraTS 2019 dataset. Data can be downloaded from http://braintumorsegmentation.org/
