import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision as tv
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from numpy.random import RandomState
from torch import distributed as dist
from torch import nn, optim
from torch.multiprocessing import Process
from torch.nn.functional import interpolate, pixel_shuffle, pixel_unshuffle, softplus
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from EMA import EMA
from ldvae import AutoencoderKL, configs
from score_sde.models.discriminator import Discriminator_large, Discriminator_small
from score_sde.models.ncsnpp_generator_adagn import NCSNpp

sys.path.append("optimizers/shampoo")
from shampoo import Shampoo
from shampoo_utils import GraftingType

_F = 16  # https://ommer-lab.com/files/latent-diffusion/kl-f32.zip
VAE = lambda: AutoencoderKL(**configs[_F], ckpt_path=f"kl-f{_F}.ckpt").eval().requires_grad_(False)
_C = {32: 64, 16: 16, 8: 4, 4: 3}


class ModeWrap:
    def __init__(self, x) -> None:
        self.x = x

    def mode(self):
        return self.x


class PixelShuffler(torch.nn.Module):
    def __init__(self, r) -> None:
        super().__init__()
        self.r = r

    def encode(self, x):
        return ModeWrap(pixel_unshuffle(x, self.r))

    def decode(self, x):
        return pixel_shuffle(x, self.r)


@torch.inference_mode()
def prepare_autoencoded_dataset(dataset, image_size, batch_size, new_shape):
    files = sum([glob(f"{dataset}/*{ext}") for ext in tv.datasets.folder.IMG_EXTENSIONS], [])
    np.random.shuffle(files)
    cache_path = f"/HDDs/{Path(dataset).stem}_ffcv_dataset.beton"

    uint8 = isinstance(VAE(), PixelShuffler)

    class ToCudaFloat(torch.nn.Module):
        def forward(self, x):
            x = x.float()
            if uint8:
                x = x.div(127.5).sub(1)
            return x.float().cuda()

    construct_loader = lambda: Loader(
        fname=cache_path,
        batch_size=batch_size,
        num_workers=24,
        os_cache=True,
        order=OrderOption.QUASI_RANDOM,
        pipelines={"image": [NDArrayDecoder(), ToTensor(), ToCudaFloat()]},
    )

    try:
        data_loader = construct_loader()
        rebuild = False
    except:
        rebuild = True

    if rebuild:

        autoencoder = VAE()

        if isinstance(autoencoder, PixelShuffler):
            transforms = tv.transforms.Compose(
                [tv.transforms.Resize(image_size, antialias=True), tv.transforms.CenterCrop(image_size)]
            )

            class IntoFFCV(Dataset):
                def __len__(self):
                    return len(files)

                def __getitem__(self, idx):
                    im = PIL.Image.open(files[idx]).convert("RGB")
                    im = torch.from_numpy(np.asarray(transforms(im))).permute(2, 0, 1).unsqueeze(0)
                    im = autoencoder.encode(im).mode()
                    return im.numpy().astype(np.uint8)

            dtype = np.dtype("uint8")
            uint8 = True

        else:
            transforms = tv.transforms.Compose(
                [
                    tv.transforms.Resize(image_size, antialias=True),
                    tv.transforms.CenterCrop(image_size),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

            class IntoAE(Dataset):
                def __len__(self):
                    return len(files)

                def __getitem__(self, idx):
                    return transforms(PIL.Image.open(files[idx]).convert("RGB"))

            autoencoder = autoencoder.cuda()

            latims = []
            for batch in tqdm(
                DataLoader(IntoAE(), batch_size=3, num_workers=torch.multiprocessing.cpu_count()),
                desc="Encoding images with VAE...",
            ):
                for img in autoencoder.encode(batch.cuda()).mode():
                    latims.append(img.unsqueeze(0).cpu().numpy())

            class IntoFFCV(Dataset):
                def __len__(self):
                    return len(latims)

                def __getitem__(self, idx):
                    return latims[idx]

            dtype = np.dtype("float32")
            uint8 = False

        print("Preprocessing latent images into FFCV dataset...")
        pipelines = {"image": NDArrayField(shape=new_shape, dtype=dtype)}
        DatasetWriter(cache_path, pipelines).from_indexed_dataset(IntoFFCV())
        print("Done!\n")

        data_loader = construct_loader()

    return infiniter(data_loader)


def infiniter(loader):
    while True:
        for batch in loader:
            yield batch


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def get_sigma_schedule(n_timestep, beta_min, beta_max, use_geometric, device):
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small

    if use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class DiffusionCoefficients:
    def __init__(self, n_timestep, beta_min, beta_max, use_geometric, device):

        self.sigmas, self.a_s, _ = get_sigma_schedule(n_timestep, beta_min, beta_max, use_geometric, device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


class PosteriorCoefficients:
    def __init__(self, n_timestep, beta_min, beta_max, use_geometric, device):

        _, _, self.betas = get_sigma_schedule(n_timestep, beta_min, beta_max, use_geometric, device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = 1 - (t == 0).type(torch.float32)

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, nz, latents=None):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((len(x),), i, dtype=torch.int64, device=x.device)

            if latents is None:
                latent_z = torch.randn(len(x), nz, device=x.device)
            else:
                latent_z = latents[:, i]

            x_0 = generator(x, t, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
    return x


def train(rank, gpu, args):
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device(f"cuda:{gpu}")

    if args.pixel_shuffle:
        global VAE
        VAE = lambda: PixelShuffler(args.pixel_shuffle)
        new_shape = (
            args.num_channels * args.pixel_shuffle ** 2,
            args.image_size // args.pixel_shuffle,
            args.image_size // args.pixel_shuffle,
        )
    else:
        new_shape = _C[_F], args.image_size // _F, args.image_size // _F

    train_loader = prepare_autoencoded_dataset(args.dataset, args.image_size, args.batch_size, new_shape)
    args.num_channels, args.image_size, args.image_size = new_shape

    bs = args.batch_size
    n_t = args.num_timesteps
    n_c = args.num_channels
    n_z = args.nz  # latent dimension
    im_size = args.image_size

    netG = NCSNpp(
        image_size=args.image_size,
        num_channels=args.num_channels,
        nz=args.nz,
        z_emb_dim=args.z_emb_dim,
        n_mlp=args.n_mlp,
        num_channels_dae=args.num_channels_dae,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        attn_resolutions=args.attn_resolutions,
        not_use_tanh=args.not_use_tanh,
        dropout=args.dropout,
        resamp_with_conv=args.resamp_with_conv,
        conditional=args.conditional,
        fir=args.fir,
        fir_kernel=args.fir_kernel,
        skip_rescale=args.skip_rescale,
        resblock_type=args.resblock_type,
        progressive=args.progressive,
        progressive_input=args.progressive_input,
        progressive_combine=args.progressive_combine,
        embedding_type=args.embedding_type,
        fourier_scale=args.fourier_scale,
    ).to(device)
    netD = (Discriminator_small if args.image_size < 256 else Discriminator_large)(
        nc=2 * n_c, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)
    ).to(device)

    # broadcast_params(netG.parameters())
    # broadcast_params(netD.parameters())

    if args.shampoo:
        optimizerD = Shampoo(
            netD.parameters(),
            lr=args.lr_d,
            betas=(args.beta1, args.beta2),
            grafting_type=GraftingType.ADAM,
            grafting_epsilon=1e-08,
            grafting_beta2=args.beta2,
            root_inv_dist=False,
        )
        optimizerG = Shampoo(
            netG.parameters(),
            lr=args.lr_g,
            betas=(args.beta1, args.beta2),
            grafting_type=GraftingType.ADAM,
            grafting_epsilon=1e-08,
            grafting_beta2=args.beta2,
            root_inv_dist=False,
        )
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.kimg, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.kimg, eta_min=1e-5)

    # netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    # netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    coeff = DiffusionCoefficients(n_t, args.beta_min, args.beta_max, args.use_geometric, device)
    pos_coeff = PosteriorCoefficients(n_t, args.beta_min, args.beta_max, args.use_geometric, device)

    exp = args.exp
    parent_dir = f"{args.exp_dir}/{Path(args.dataset).stem}"
    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            shutil.copyfile(__file__, os.path.join(exp_path, os.path.basename(__file__)))
            shutil.copytree("score_sde/models", os.path.join(exp_path, "score_sde/models"))

    if args.resume:
        checkpoint_file = os.path.join(exp_path, "resume_state.pth")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        init_iteration = checkpoint["nimgs"]
        netG.load_state_dict(checkpoint["netG_dict"])
        optimizerG.load_state_dict(checkpoint["optimizerG"])
        schedulerG.load_state_dict(checkpoint["schedulerG"])
        netD.load_state_dict(checkpoint["netD_dict"])
        optimizerD.load_state_dict(checkpoint["optimizerD"])
        schedulerD.load_state_dict(checkpoint["schedulerD"])
        print(f"=> loaded checkpoint ({init_iteration / 1000:.2f} kimg)")
    else:
        init_iteration = 0

    with tqdm(
        range(init_iteration, args.kimg * 1000, bs), desc="Training latent diffusion GAN...", unit_scale=bs, unit="img"
    ) as pbar:
        for iteration in pbar:
            x = next(train_loader)[0]

            # D STEP
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(0, n_t, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            if args.lazy_reg is None or iteration % (bs * args.lazy_reg) == 0:
                x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)

            errD_real = softplus(-D_real)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None or iteration % (bs * args.lazy_reg) == 0:
                grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(bs, n_z, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errD_fake = softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            optimizerD.step()

            # G STEP
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, n_t, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(bs, n_z, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errG = softplus(-output)
            errG = errG.mean()
            errG.backward()
            optimizerG.step()

            if not args.no_lr_decay:
                schedulerG.step()
                schedulerD.step()

            if rank == 0 and iteration % (bs * 100) == 0:
                pbar.write(
                    f"kimg {iteration / 1000:.2f}, G Loss: {errG.item():.4f}, D Loss: {(errD_real + errD_fake).item():.4f}"
                )

            if rank == 0 and (
                (iteration % (args.save_image_every * 1000) < bs) or (iteration % 1000 < bs and iteration < 10_000)
            ):
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                with torch.inference_mode():
                    x_t_1 = torch.from_numpy(RandomState(42).randn(16 * 9, n_c, im_size, im_size)).float()
                    z = torch.from_numpy(RandomState(42).randn(16 * 9, n_t, n_z)).float()

                    autoencoder = VAE().cuda()

                    imgs = []
                    for b in range(0, 16 * 9, bs):
                        samples = sample_from_model(
                            pos_coeff, netG, n_t, x_t_1[b : b + bs].to(device), n_z, z[b : b + bs].to(device)
                        )
                        for sample in samples:
                            imgs.append(autoencoder.decode(sample.unsqueeze(0)).cpu())
                    imgs = torch.cat(imgs)
                    if args.image_size * 32 > 512:
                        imgs = interpolate(imgs, (512, 512), mode="bilinear", align_corners=True)
                    tv.utils.save_image(
                        imgs,
                        f"{exp_path}/samples_{Path(args.dataset).stem}_kimg{round(iteration / 1000)}.jpg",
                        nrow=16,
                        normalize=True,
                    )

                    del autoencoder
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

            if rank == 0 and iteration % (args.save_ckpt_every * 1000) < bs:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(
                    netG.state_dict(),
                    os.path.join(exp_path, f"netG_{Path(args.dataset).stem}_kimg{round(iteration / 1000)}.pth"),
                )
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

            if rank == 0 and iteration % (args.save_state_every * 1000) < bs:
                torch.save(
                    {
                        "nimgs": iteration,
                        "args": args,
                        "netG_dict": netG.state_dict(),
                        "optimizerG": optimizerG.state_dict(),
                        "schedulerG": schedulerG.state_dict(),
                        "netD_dict": netD.state_dict(),
                        "optimizerD": optimizerD.state_dict(),
                        "schedulerD": schedulerD.state_dict(),
                    },
                    os.path.join(exp_path, "resume_state.pth"),
                )


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = "6020"
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# fmt:off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022_05_24, help="seed used for initialization")

    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--image_size", type=int, default=1024, help="size of image")
    parser.add_argument("--num_channels", type=int, default=3, help="channel of image")
    parser.add_argument("--pixel_shuffle", type=int, default=4, help="pixel shuffling factor")
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--use_geometric", action="store_true", default=False)
    parser.add_argument("--beta_min", type=float, default=0.1, help="beta_min for diffusion")
    parser.add_argument("--beta_max", type=float, default=20.0, help="beta_max for diffusion")

    parser.add_argument("--num_channels_dae", type=int, default=128, help="number of initial channels in denosing model")
    parser.add_argument("--n_mlp", type=int, default=4, help="number of mlp layers for z")
    parser.add_argument("--ch_mult", default=[1, 2, 2, 4], nargs="+", type=int, help="channel multiplier")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of resnet blocks per scale")
    parser.add_argument("--attn_resolutions", default=[16], type=int, nargs="*", help="resolution of applying attention")
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--resamp_with_conv", action="store_false", default=True, help="always up/down sampling with conv")
    parser.add_argument("--conditional", action="store_false", default=True, help="noise conditional")
    parser.add_argument("--fir", action="store_false", default=True, help="FIR")
    parser.add_argument("--fir_kernel", default=[1, 3, 3, 1], help="FIR kernel")
    parser.add_argument("--skip_rescale", action="store_false", default=True, help="skip rescale")
    parser.add_argument("--resblock_type", default="biggan", help="style of resnet block, choice in biggan and ddpm")
    parser.add_argument("--progressive", type=str, default="none", choices=["none", "output_skip", "residual"], help="progressive type for output")
    parser.add_argument("--progressive_input", type=str, default="residual", choices=["none", "input_skip", "residual"], help="progressive type for input")
    parser.add_argument("--progressive_combine", type=str, default="sum", choices=["sum", "cat"], help="progressive combine method.")

    parser.add_argument("--embedding_type", type=str, default="positional", choices=["positional", "fourier"], help="type of time embedding")
    parser.add_argument("--fourier_scale", type=float, default=16.0, help="scale of fourier transform")
    parser.add_argument("--not_use_tanh", action="store_false", default=True)

    # generator and training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--exp_dir", default="/home/hans/modelzoo/diffusionGAN/", help="directory to save experiments in")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=2)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--t_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--kimg", type=int, default=16_000)
    parser.add_argument("--ngf", type=int, default=64)

    parser.add_argument("--lr_g", type=float, default=1.5e-4, help="learning rate g")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate d")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)
    parser.add_argument("--shampoo", action="store_true", help="Use Shampoo pre-conditioner for Adam")

    parser.add_argument("--use_ema", action="store_false", default=True, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")

    parser.add_argument("--r1_gamma", type=float, default=0.05, help="coef for r1 reg")
    parser.add_argument("--lazy_reg", type=int, default=12, help="lazy regularization")

    parser.add_argument("--save_image_every", type=int, default=10, help="save image samples every x kimg")
    parser.add_argument("--save_ckpt_every", type=int, default=60, help="save ckpt every x kimg")
    parser.add_argument("--save_state_every", type=int, default=1, help="save full state for resuming every x kimg")

    # ddp
    parser.add_argument("--num_proc_node", type=int, default=1, help="The number of nodes in multi node env.")
    parser.add_argument("--num_process_per_node", type=int, default=1, help="number of gpus")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print("Node rank %d, local proc %d, global proc %d" % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train(0, 0, args)
