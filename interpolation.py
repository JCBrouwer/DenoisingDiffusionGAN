import argparse
import os
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.nn.functional import conv1d, pad
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import tqdm

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from test_ddgan import Posterior_Coefficients, extract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_filter(x, sigma, mode="circular"):
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4), len(x))
    sigma = radius / 4
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma**2 * kernel**2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    x = pad(x, (radius, radius), mode=mode)
    x = conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x


def spline_loop(y, size):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, 1, size).to(y)
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


@torch.inference_mode()
def interpolate(
    # output settings
    n_interps,
    n_frames,
    num_channels,
    image_size,
    batch_size,
    seed,
    num_timesteps,
    init_noise_smooth,
    latent_smooth,
    post_noise_smooth,
    # checkpoint settings
    dataset,
    exp,
    nz,
    epoch_id,
    # posterior settings
    use_geometric,
    beta_min,
    beta_max,
    **kwargs,
):
    if seed is None:
        seed = np.random.randint(0, 2**16)
    torch.manual_seed(seed)
    np.random.seed(seed)

    netG = NCSNpp(argparse.Namespace(**{**locals(), **kwargs})).to(device)

    parent_dir = f"/home/hans/modelzoo/dd_gan/{Path(dataset).stem}"
    exp_path = os.path.join(parent_dir, exp)
    ckpt = torch.load(f"{exp_path}/netG_{epoch_id}.pth", map_location=device)

    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    C = Posterior_Coefficients(num_timesteps, beta_min, beta_max, use_geometric, device)
    post_mu1 = C.posterior_mean_coef1
    post_mu2 = C.posterior_mean_coef2
    post_log_var = C.posterior_log_variance_clipped

    for i in range(n_interps):
        init_noise = np.random.RandomState(seed + i + 1).randn(n_frames, num_channels, image_size, image_size)
        init_noise = torch.from_numpy(init_noise).float()
        init_noise = gaussian_filter(init_noise, init_noise_smooth)
        init_noise /= init_noise.square().mean().sqrt()

        post_noise = np.random.RandomState(seed + i + 2).randn(n_frames, num_channels, image_size, image_size)
        post_noise = torch.from_numpy(post_noise).float()
        post_noise = gaussian_filter(post_noise, post_noise_smooth)
        post_noise /= post_noise.square().mean().sqrt()

        latents = np.random.RandomState(seed + i + 3).randn(n_frames, num_timesteps, nz)
        latents = torch.from_numpy(latents).float()
        latents = gaussian_filter(latents, latent_smooth)
        latents /= latents.square().mean().sqrt()

        imgs = []
        for b in tqdm(range(0, n_frames, batch_size)):

            x_t = init_noise[b : b + batch_size].to(device)
            p_n = post_noise[b : b + batch_size].to(device)

            for i in reversed(range(num_timesteps)):
                t = torch.full((x_t.size(0),), i, dtype=torch.int64, device=device)
                z = latents[b : b + batch_size, i].to(device)

                x_0 = netG(x_t, t, z)

                x_t = extract(post_mu1, t, x_t.shape) * x_0 + extract(post_mu2, t, x_t.shape) * x_t
                if i != 0:
                    log_var = extract(post_log_var, t, x_t.shape)
                    x_t += torch.exp(0.5 * log_var) * p_n

            imgs.append(x_t.cpu())

        filename = f"{exp_path}/interpolation_netG_{epoch_id}_seed{seed}_{str(uuid4())[:6]}.mp4"
        video = torch.cat(imgs).permute(0, 2, 3, 1).add(1).div(2).mul(255).cpu()
        torchvision.io.write_video(filename, video, fps=24)


if __name__ == "__main__":
    # fmt:off
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=None, help="seed used for initialization")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--epoch_id", type=int, default=1000)
    parser.add_argument("--num_channels", type=int, default=3, help="channel of image")
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--use_geometric", action="store_true", default=False)
    parser.add_argument("--beta_min", type=float, default=0.1, help="beta_min for diffusion")
    parser.add_argument("--beta_max", type=float, default=20.0, help="beta_max for diffusion")

    parser.add_argument("--num_channels_dae", type=int, default=128, help="number of initial channels in denosing model")
    parser.add_argument("--n_mlp", type=int, default=3, help="number of mlp layers for z")
    parser.add_argument("--ch_mult", nargs="+", type=int, help="channel multiplier")

    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of resnet blocks per scale")
    parser.add_argument("--attn_resolutions", default=(16,), help="resolution of applying attention")
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--resamp_with_conv", action="store_false", default=True, help="always up/down sampling with conv")
    parser.add_argument("--conditional", action="store_false", default=True, help="noise conditional")
    parser.add_argument("--fir", action="store_false", default=True, help="FIR")
    parser.add_argument("--fir_kernel", default=[1, 3, 3, 1], help="FIR kernel")
    parser.add_argument("--skip_rescale", action="store_false", default=True, help="skip rescale")
    parser.add_argument("--resblock_type", default="biggan", help="tyle of resnet block, choice in biggan and ddpm")
    parser.add_argument("--progressive", type=str, default="none", choices=["none", "output_skip", "residual"], help="progressive type for output")
    parser.add_argument("--progressive_input", type=str, default="residual", choices=["none", "input_skip", "residual"], help="progressive type for input")
    parser.add_argument("--progressive_combine", type=str, default="sum", choices=["sum", "cat"], help="progressive combine method.")

    parser.add_argument("--embedding_type", type=str, default="positional", choices=["positional", "fourier"], help="type of time embedding")
    parser.add_argument("--fourier_scale", type=float, default=16.0, help="scale of fourier transform")
    parser.add_argument("--not_use_tanh", action="store_true", default=False)

    # generator and training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--real_img_dir", default="./pytorch_fid/cifar10_train_stat.npy", help="directory to real images for FID computation")

    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--image_size", type=int, default=32, help="size of image")

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=4)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--t_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32, help="sample generating batch size")

    parser.add_argument("--n_frames", type=int, default=20*24, help="number of frames in each interpolation")
    parser.add_argument("--n_interps", type=int, default=1, help="number of interpolation videos to generate")
    parser.add_argument("--init_noise_smooth", type=int, default=30, help="sigma of temporal gaussian filter for initial noise")
    parser.add_argument("--latent_smooth", type=int, default=400, help="sigma of temporal gaussian filter for latent vectors")
    parser.add_argument("--post_noise_smooth", type=int, default=400, help="sigma of temporal gaussian filter for posterior noise")

    interpolate(**vars(parser.parse_args()))
