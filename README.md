# Pretraining Deformable Image Registration Networks with Random Images

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

keywords: Image Registration, Self-supervised Learning, Pretraining

This is a **PyTorch** implementation of my paper:

[Chen, Junyu, et al. "Pretraining Deformable Image Registration Networks with Random Images." Medical Imaging with Deep Learning. 2025.](https://openreview.net/forum?id=NJANlZzxfi)

The core idea of this work is to leverage randomly generated images to initialize (or pretrain) the encoder of an image registration network. To achieve this, we designed temporary lightweight decoders that are attached to the encoder of the registration DNN, and pretrained the resulting network using a standard image registration loss function on the task of aligning pairs of random images.\
This approach is implemented in the [MIR package](https://github.com/junyuchen245/MIR). The source code for generating random images can be found [here](https://github.com/junyuchen245/MIR/tree/main/MIR/random_image_generation), and the lightweight decoder is implemented [here](https://github.com/junyuchen245/MIR/blob/main/MIR/models/Selfsupervised_Learning_Heads.py). The repository also includes training and inference scripts to reproduce the results reported in the paper.

## Pretraining and Fine-tuning Pipeline
### Step 1: Pretraining the encoder on a proxy task of registering random images
Run `python -u train_SSL.py` to initiate the pretraining. We first extract the encoder from the registration DNN and connect it to a lightweight decoder for pretraining.
https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs/blob/88a330b9b26313a25d346725df0464cfcdc32968/scripts/train_SSL.py#L36-L45
In each iteration, a pair of random images is generated using `data=rs.gen_shapes(.)`, in which `data[0]` and `data[1]` contains moving and fixed random images along with their binary label maps stored in `data[2]` and `data[3]`.
https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs/blob/88a330b9b26313a25d346725df0464cfcdc32968/scripts/train_SSL.py#L84-L86
We then simply compute the registration loss for pretraining.
https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs/blob/88a330b9b26313a25d346725df0464cfcdc32968/scripts/train_SSL.py#L105-L106
### Step 2: Fine-tuning the DNN on a downstream registration task
TBA

## Pretraining Strategy Overview
<img src="https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs/blob/main/figs/overview.jpg" width="800"/>

## Generating Paired Random Images
<img src="https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs/blob/main/figs/gen_images.jpg" width="500"/>

## Pretraining Reduces Amount of Data Needed to Achieve Competitive Performance
<img src="https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs/blob/main/figs/Reducing_Data.jpg" width="800"/>

