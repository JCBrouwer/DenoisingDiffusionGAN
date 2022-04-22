import numpy as np
import torch

from ldvae import AutoencoderKL, configs
from score_sde.models.ncsnpp_generator_adagn import NCSNpp


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def get_sigma_schedule(num_timesteps, beta_min, beta_max, use_geometric, device):
    n_timestep = num_timesteps
    beta_min = beta_min
    beta_max = beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small

    if use_geometric:
        var = beta_min * ((beta_max / beta_min) ** t)
    else:
        log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class PosteriorCoefficients:
    def __init__(self, num_timesteps, beta_min, beta_max, use_geometric, device="cpu"):

        _, _, self.betas = get_sigma_schedule(num_timesteps, beta_min, beta_max, use_geometric, device)

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


class DiffusionGAN(torch.nn.Module):
    def __init__(
        self,
        ckpt=None,
        latent=True,
        num_timesteps=4,
        image_size=32,
        num_channels=64,
        nz=100,
        z_emb_dim=256,
        n_mlp=4,
        num_channels_dae=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[4, 8, 16, 32],
        not_use_tanh=True,
        conditional=True,
        kl_f=32,
    ):
        super().__init__()

        self.latent = latent
        self.num_timesteps = num_timesteps
        self.G = NCSNpp(
            image_size=image_size,
            num_channels=num_channels,
            nz=nz,
            z_emb_dim=z_emb_dim,
            n_mlp=n_mlp,
            num_channels_dae=num_channels_dae,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            not_use_tanh=not_use_tanh,
            conditional=conditional,
        )
        if ckpt is not None:
            self.G.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.G.eval()

        if latent:
            self.D = AutoencoderKL(**configs[kl_f], ckpt_path=f"kl-f{kl_f}.ckpt").eval().requires_grad_(False)

        C = PosteriorCoefficients(num_timesteps, beta_min=0.1, beta_max=20.0, use_geometric=False)
        self.register_buffer("post_mu1", C.posterior_mean_coef1)
        self.register_buffer("post_mu2", C.posterior_mean_coef2)
        self.register_buffer("post_log_var", C.posterior_log_variance_clipped)

    def forward(self, x_t, p_n, zs):
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((len(x_t),), i, dtype=torch.int64, device=x_t.device)

            # get proposal image
            x_0 = self.G(x_t, t, zs[:, i])

            # weigh versus current image
            x_t = extract(self.post_mu1, t, x_t.shape) * x_0 + extract(self.post_mu2, t, x_t.shape) * x_t

            # add noise to proposal image for next step (unless there is no next step)
            if i > 0:
                log_var = extract(self.post_log_var, t, x_t.shape)
                x_t += torch.exp(0.5 * log_var) * p_n

        if self.latent:
            x_t = self.D.decode(x_t)

        return x_t.clamp(-1, 1)
