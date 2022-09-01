# DenoisingDiffusionGAN

My fork of [denoising-diffusion-gan](github.com/NVlabs/denoising-diffusion-gan).

There's a new training script which can train a diffusion GAN in the latent space of a VAE or use pixel shuffle to generate 1024px images with essentially the same training memory requirements. The results weren't stunning immediately, but maybe with some better hyperparameters and more training time this could work pretty well.

There's also a script for generating video interpolations: `interpolate.py`. This supports video initialization, latent interpolation, and arbitrary resolution generation (VRAM permitting).

```bash
optional arguments:
  -h, --help            show this help message and exit
  
  # Interpolation parameters
  --seed SEED           seed used for initialization
  --n_frames N_FRAMES   number of frames in each interpolation
  --fps FPS             frames per second in output video
  --n_interps N_INTERPS
                        number of interpolation videos to generate
  --video_init VIDEO_INIT
                        video to use as initialization
  --var_scale VAR_SCALE
                        weight of init noise when video_init is used. lower values preserve content video more.
  --init_noise_smooth INIT_NOISE_SMOOTH
                        sigma of temporal gaussian filter for initial noise
  --latent_smooth LATENT_SMOOTH
                        sigma of temporal gaussian filter for latent vectors
  --post_noise_smooth POST_NOISE_SMOOTH
                        sigma of temporal gaussian filter for posterior noise
  --interp_seeds [INTERP_SEEDS ...]
                        seeds for spline interpolation
  --overscaling OVERSCALING
                        factor with which to increase image size (relative to training size)
  --smooth_device SMOOTH_DEVICE
                        what device to perform smoothing on ('cpu' for long/large/high-sigma interpolations)
  --init_video_smooth INIT_VIDEO_SMOOTH
                        how much to smoothen initialization video
  --out_dir OUT_DIR
  
  # Model-related parameters
  --ckpt CKPT
  --image_size IMAGE_SIZE
                        size of image
  --num_channels NUM_CHANNELS
                        channel of image
  --nz NZ
  --num_timesteps NUM_TIMESTEPS
  --batch_size BATCH_SIZE
                        sample generating batch size
  --n_mlp N_MLP         number of mlp layers for z
  --ch_mult CH_MULT [CH_MULT ...]
                        channel multiplier
  --num_channels_dae NUM_CHANNELS_DAE
                        number of initial channels in denosing model
  --num_res_blocks NUM_RES_BLOCKS
                        number of resnet blocks per scale
  --attn_resolutions [ATTN_RESOLUTIONS ...]
                        resolution of applying attention
  --not_use_tanh
  --latent              Model is a LatentDiffusionGAN
```


# Official PyTorch implementation of "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs" [(ICLR 2022 Spotlight Paper)](https://arxiv.org/abs/2112.07804) #

<div align="center">
  <a href="https://xavierxiao.github.io/" target="_blank">Zhisheng&nbsp;Xiao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://karstenkreis.github.io/" target="_blank">Karsten&nbsp;Kreis</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://latentspace.cc/arash_vahdat/" target="_blank">Arash&nbsp;Vahdat</a>
  <br> <br>
  <a href="https://nvlabs.github.io/denoising-diffusion-gan/" target="_blank">Project&nbsp;Page</a>
</div>
<br>
<br>

<div align="center">
    <img width="800" alt="teaser" src="assets/teaser.png"/>
</div>

Generative denoising diffusion models typically assume that the denoising distribution can be modeled by a Gaussian distribution. This assumption holds only for small denoising steps, which in practice translates to thousands of denoising steps in the synthesis process. In our denoising diffusion GANs, we represent the denoising model using multimodal and complex conditional GANs, enabling us to efficiently generate data in as few as two steps.

## Set up datasets ##
We trained on several datasets, including CIFAR10, LSUN Church Outdoor 256 and CelebA HQ 256. 
For large datasets, we store the data in LMDB datasets for I/O efficiency. Check [here](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) for information regarding dataset preparation.


## Training Denoising Diffusion GANs ##
We use the following commands on each dataset for training denoising diffusion GANs.

#### CIFAR-10 ####

We train Denoising Diffusion GANs on CIFAR-10 using 4 32-GB V100 GPU. 
```
python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 4 \
--ch_mult 1 2 2 2 --save_content
```

#### LSUN Church Outdoor 256 ####

We train Denoising Diffusion GANs on LSUN Church Outdoor 256 using 8 32-GB V100 GPU. 
```
python3 train_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 8 --save_content
```

#### CelebA HQ 256 ####

We train Denoising Diffusion GANs on CelebA HQ 256 using 8 32-GB V100 GPUs. 
```
python3 train_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
--num_res_blocks 2 --batch_size 4 --num_epoch 800 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10  --num_process_per_node 8 --save_content
```

## Pretrained Checkpoints ##
We have released pretrained checkpoints on CIFAR-10 and CelebA HQ 256 at this 
[Google drive directory](https://drive.google.com/drive/folders/1UkzsI0SwBRstMYysRdR76C1XdSv5rQNz?usp=sharing).
Simply download the `saved_info` directory to the code directory. Use `--epoch_id 1200` for CIFAR-10 and `--epoch_id 550`
for CelebA HQ 256 in the commands below.

## Evaluation ##
After training, samples can be generated by calling ```test_ddgan.py```. We evaluate the models with single V100 GPU.
Below, we use `--epoch_id` to specify the checkpoint saved at a particular epoch.
Specifically, for models trained by above commands, the scripts for generating samples on CIFAR-10 is
```
python3 test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id $EPOCH
```
The scripts for generating samples on CelebA HQ is 
```
python3 test_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id $EPOCH
```
The scripts for generating samples on LSUN Church Outdoor is 
```
python3 test_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id $EPOCH
```

We use the [PyTorch](https://github.com/mseitzer/pytorch-fid) implementation to compute the FID scores, and in particular, codes for computing the FID are adapted from [FastDPM](https://github.com/FengNiMa/FastDPM_pytorch).

To compute FID, run the same scripts above for sampling, with additional arguments ```--compute_fid``` and ```--real_img_dir /path/to/real/images```.

For Inception Score, save samples in a single numpy array with pixel values in range [0, 255] and simply run 
```
python ./pytorch_fid/inception_score.py --sample_dir /path/to/sampled_images
```
where the code for computing Inception Score is adapted from [here](https://github.com/tsc2017/Inception-Score).

For Improved Precision and Recall, follow the instruction [here](https://github.com/kynkaat/improved-precision-and-recall-metric).


## License ##
Please check the LICENSE file. Denoising diffusion GAN may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

## Bibtex ##
Cite our paper using the following bibtex item:

```
@inproceedings{
xiao2022tackling,
title={Tackling the Generative Learning Trilemma with Denoising Diffusion GANs},
author={Zhisheng Xiao and Karsten Kreis and Arash Vahdat},
booktitle={International Conference on Learning Representations},
year={2022}
}
```

## Contributors ##
Denoising Diffusion GAN was built primarily by [Zhisheng Xiao](https://xavierxiao.github.io/) during a summer 
internship at NVIDIA research.
