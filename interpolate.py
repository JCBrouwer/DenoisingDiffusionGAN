import argparse
import os
from pathlib import Path
from uuid import uuid4

import decord as de
import numpy as np
import torch
import torchvision
from torch.nn.functional import conv1d, interpolate, pad
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import tqdm

from diffusionGAN import DiffusionGAN

de.bridge.set_bridge("torch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_filter(x, sigma, mode="circular"):
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    request_vs_reality = int(sigma * 4) - len(x)
    if request_vs_reality > 0:
        radius = len(x)
        sigma = radius / 4
        repeat = 1 + request_vs_reality // len(x)
    else:
        radius = int(sigma * 4)
        repeat = 1

    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    for _ in range(repeat):
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
    batch_size,
    # interpolation settings
    seed,
    interp_seeds,
    n_frames,
    overscaling,
    fps,
    n_interps,
    video_init,
    var_scale,
    init_noise_smooth,
    latent_smooth,
    post_noise_smooth,
    smooth_device,
    init_video_smooth,
    # checkpoint settings
    image_size,
    num_channels,
    num_timesteps,
    nz,
    n_mlp,
    ckpt,
    out_dir,
    **model_kwargs
):
    if seed is None:
        seed = np.random.randint(0, 2 ** 16)
    torch.manual_seed(seed)
    np.random.seed(seed)

    G = DiffusionGAN(
        ckpt=ckpt,
        latent=True,
        num_timesteps=num_timesteps,
        image_size=image_size,
        num_channels=num_channels,
        nz=nz,
        n_mlp=n_mlp,
        **model_kwargs,
    ).to(device)

    if video_init is not None:
        v = de.VideoReader(video_init, width=round(image_size * overscaling), height=round(image_size * overscaling))
        fps = round(v.get_avg_fps())
        video = v[:].permute(0, 3, 1, 2).float().contiguous()
        video = gaussian_filter(video.to(smooth_device), init_video_smooth)
        video -= video.mean()
        video /= video.std()
        n_frames = len(video)
    else:
        video = None

    for j in range(n_interps):
        if interp_seeds is None:
            latents = np.random.RandomState(seed + j).randn(n_frames, num_timesteps, nz)
            latents = torch.from_numpy(latents).float()
        else:
            latent_selection = [np.random.RandomState(s).randn(1, num_timesteps, nz) for s in interp_seeds]
            latent_selection = torch.cat([torch.from_numpy(l).float() for l in latent_selection])
            latents = spline_loop(latent_selection, n_frames)
        if n_frames > 1:
            latents = gaussian_filter(latents.to(smooth_device), latent_smooth)
            latents /= latents.square().mean().sqrt()
            latents = latents.cpu()

        init_noise = np.random.RandomState(seed + j + 1).randn(
            n_frames, num_channels, round(image_size * overscaling), round(image_size * overscaling)
        )
        init_noise = torch.from_numpy(init_noise).float()
        if n_frames > 1:
            init_noise = gaussian_filter(init_noise.to(smooth_device), init_noise_smooth)
            init_noise /= init_noise.square().mean().sqrt()
            init_noise = init_noise.cpu()

        post_noise = np.random.RandomState(seed + j + 2).randn(
            n_frames, num_channels, round(image_size * overscaling), round(image_size * overscaling)
        )
        post_noise = torch.from_numpy(post_noise).float()
        if n_frames > 1:
            post_noise = gaussian_filter(post_noise.to(smooth_device), post_noise_smooth)
            post_noise /= post_noise.square().mean().sqrt()
            post_noise = post_noise.cpu()

        if video is None:
            init = init_noise
        else:
            init = (video + var_scale * init_noise) / (var_scale + 1)

        imgs = []
        for b in tqdm(range(0, n_frames, batch_size)):
            x_t = init[b : b + batch_size].to(device)
            p_n = post_noise[b : b + batch_size].to(device)
            zs = latents[b : b + batch_size].to(device)
            x_t = G(x_t, p_n, zs)
            imgs.append(x_t.cpu())

        if n_frames > 1:
            task = f"{Path(video_init).stem}_transfer" if video_init is not None else "interpolation"
            filename = f"{out_dir}/diffusionGAN_{task}_{Path(ckpt).stem}_seed{seed + j}_{str(uuid4())[:6]}.mp4"
            output = torch.cat(imgs).permute(0, 2, 3, 1).add(1).div(2).mul(255).cpu()
            torchvision.io.write_video(filename, output, fps=fps, options={"crf": "10"})

        else:
            filename = f"{out_dir}/{Path(ckpt).stem}_seed{seed + j}_{str(uuid4())[:6]}.jpg"
            output = imgs[0].squeeze().add(1).div(2).mul(255).round().byte().cpu()
            torchvision.io.write_jpeg(output, filename, quality=95)


if __name__ == "__main__":
    # fmt:off
    parser = argparse.ArgumentParser("ddgan parameters")

    parser.add_argument("--ckpt", type=str)

    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--num_channels", type=int, default=64, help="channel of image")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32, help="sample generating batch size")
    parser.add_argument("--n_mlp", type=int, default=4, help="number of mlp layers for z")
    parser.add_argument("--ch_mult", default=[1, 2, 2, 4], nargs="+", type=int, help="channel multiplier")
    parser.add_argument("--num_channels_dae", type=int, default=128, help="number of initial channels in denosing model")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of resnet blocks per scale")
    parser.add_argument("--attn_resolutions", default=[4, 8, 16, 32], type=int, nargs="*", help="resolution of applying attention")
    parser.add_argument("--not_use_tanh", action="store_false", default=True)

    parser.add_argument("--seed", type=int, default=None, help="seed used for initialization")
    parser.add_argument("--n_frames", type=int, default=20*24, help="number of frames in each interpolation")
    parser.add_argument("--fps", type=float, default=24, help="frames per second in output video")
    parser.add_argument("--n_interps", type=int, default=1, help="number of interpolation videos to generate")
    parser.add_argument("--video_init", type=str, default=None, help="video to use as initialization")
    parser.add_argument("--var_scale", type=float, default=1., help="weight of init noise when video_init is used. lower values preserve content video more.")
    parser.add_argument("--init_noise_smooth", type=int, default=96, help="sigma of temporal gaussian filter for initial noise")
    parser.add_argument("--latent_smooth", type=int, default=24, help="sigma of temporal gaussian filter for latent vectors")
    parser.add_argument("--post_noise_smooth", type=int, default=96, help="sigma of temporal gaussian filter for posterior noise")
    parser.add_argument("--interp_seeds", type=int, default=None, nargs="*", help="seeds for spline interpolation")
    parser.add_argument("--overscaling", type=float, default=1, help="factor with which to increase image size (relative to training size)")
    parser.add_argument("--smooth_device", type=str, default="cuda", help="what device to perform smoothing on ('cpu' for long/large/high-sigma interpolations)")
    parser.add_argument("--init_video_smooth", type=float, default=12, help="how much to smoothen initialization video")

    parser.add_argument("--out_dir", type=str, default='/home/hans/modelzoo/diffusionGAN/')

    interpolate(**vars(parser.parse_args()))
