import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class VAE_encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE_encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=4, stride=2, padding=1
        )  # (32, 50, 50)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # (64, 25, 25)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )  # (128, 13, 13)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=1
        )  # (256, 7, 7)
        self.bn4 = nn.BatchNorm2d(256)

        self.flatten_size = 256 * 7 * 7
        self.fc_pre_latent = nn.Linear(self.flatten_size, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_lov_var = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(-1, self.flatten_size)
        x = F.relu(self.bn_fc(self.fc_pre_latent(x)))

        mu = self.fc_mu(x)
        log_var = self.fc_lov_var(x)

        return mu, log_var


class VAE_decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE_decoder, self).__init__()
        self.latent_dim = latent_dim
        self.reshape_channels = 256
        self.reshape_size = 7

        self.fc_decode_start = nn.Linear(latent_dim, 1024)
        self.bn_fc_decode = nn.BatchNorm1d(1024)
        self.fc_decode_reshape = nn.Linear(
            1024, self.reshape_channels * (self.reshape_size**2)
        )
        self.bn_fc_reshape = nn.BatchNorm1d(
            self.reshape_channels * (self.reshape_size**2)
        )

        # (256, 7, 7)
        self.deconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=0
        )
        self.bn_d1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=0
        )  # (64, 25, 25)
        self.bn_d2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1, output_padding=0
        )  # (32, 50, 50)
        self.bn_d3 = nn.BatchNorm2d(32)

        self.deconv4 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1, output_padding=1
        )  # (1, 101, 101)

    def forward(self, z):
        x = F.relu(self.bn_fc_decode(self.fc_decode_start(z)))
        x = F.relu(self.bn_fc_reshape(self.fc_decode_reshape(x)))
        x = x.view(-1, self.reshape_channels, self.reshape_size, self.reshape_size)

        x = F.relu(self.bn_d1(self.deconv1(x)))
        x = F.relu(self.bn_d2(self.deconv2(x)))
        x = F.relu(self.bn_d3(self.deconv3(x)))

        reconstructed_x = torch.sigmoid(self.deconv4(x))
        return reconstructed_x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = VAE_encoder(latent_dim)
        self.decoder = VAE_decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample from standard normal distribution
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, log_var


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UNetLatent(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        latent_reshape_channels=8,
        time_emb_dim=256,
        init_unet_channels=64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_reshape_channels = latent_reshape_channels
        self.latent_reshape_side = int((latent_dim / latent_reshape_channels) ** 0.5)

        if (
            latent_reshape_channels
            * self.latent_reshape_side
            * self.latent_reshape_side
            != latent_dim
        ):
            raise ValueError(
                f"latent_dim ({latent_dim}) cannot be perfectly reshaped with "
                f"{latent_reshape_channels} channels into a square image-like tensor."
            )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # --- Layers for initial_conv block ---
        self.initial_conv_conv = nn.Conv2d(
            latent_reshape_channels,
            init_unet_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.initial_conv_bn = nn.BatchNorm2d(init_unet_channels)
        self.initial_conv_relu = nn.ReLU()
        self.initial_conv_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels), nn.ReLU()
        )

        # --- Layers for down1_conv1 block ---
        self.down1_conv1_conv = nn.Conv2d(
            init_unet_channels,
            init_unet_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down1_conv1_bn = nn.BatchNorm2d(init_unet_channels * 2)
        self.down1_conv1_relu = nn.ReLU()
        self.down1_conv1_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels * 2), nn.ReLU()
        )
        self.down1_pool = nn.Conv2d(
            init_unet_channels * 2, init_unet_channels * 2, kernel_size=2, stride=2
        )

        # --- Layers for down2_conv1 block ---
        self.down2_conv1_conv = nn.Conv2d(
            init_unet_channels * 2,
            init_unet_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down2_conv1_bn = nn.BatchNorm2d(init_unet_channels * 4)
        self.down2_conv1_relu = nn.ReLU()
        self.down2_conv1_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels * 4), nn.ReLU()
        )
        self.down2_pool = nn.Conv2d(
            init_unet_channels * 4, init_unet_channels * 4, kernel_size=2, stride=2
        )

        # --- Layers for bottleneck_conv1 block ---
        self.bottleneck_conv1_conv = nn.Conv2d(
            init_unet_channels * 4,
            init_unet_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bottleneck_conv1_bn = nn.BatchNorm2d(init_unet_channels * 4)
        self.bottleneck_conv1_relu = nn.ReLU()
        self.bottleneck_conv1_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels * 4), nn.ReLU()
        )

        # --- Layers for bottleneck_conv2 block ---
        self.bottleneck_conv2_conv = nn.Conv2d(
            init_unet_channels * 4,
            init_unet_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bottleneck_conv2_bn = nn.BatchNorm2d(init_unet_channels * 4)
        self.bottleneck_conv2_relu = nn.ReLU()
        self.bottleneck_conv2_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels * 4), nn.ReLU()
        )

        # Decoder path
        self.up1_deconv = nn.ConvTranspose2d(
            init_unet_channels * 4, init_unet_channels * 2, kernel_size=2, stride=2
        )
        # --- Layers for up1_conv1 block (input channels: init_unet_channels * 4 due to concat) ---
        self.up1_conv1_conv = nn.Conv2d(
            init_unet_channels * 4,
            init_unet_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.up1_conv1_bn = nn.BatchNorm2d(init_unet_channels * 2)
        self.up1_conv1_relu = nn.ReLU()
        self.up1_conv1_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels * 2), nn.ReLU()
        )

        self.up2_deconv = nn.ConvTranspose2d(
            init_unet_channels * 2, init_unet_channels, kernel_size=2, stride=2
        )
        # --- Layers for up2_conv1 block (input channels: init_unet_channels * 2 due to concat) ---
        self.up2_conv1_conv = nn.Conv2d(
            init_unet_channels * 2,
            init_unet_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.up2_conv1_bn = nn.BatchNorm2d(init_unet_channels)
        self.up2_conv1_relu = nn.ReLU()
        self.up2_conv1_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, init_unet_channels), nn.ReLU()
        )

        self.final_conv = nn.Conv2d(
            init_unet_channels, latent_reshape_channels, kernel_size=1
        )

    def _apply_integrated_conv_block(
        self, x, t_emb, conv_layer, bn_layer, relu_layer, time_mlp_layer
    ):
        x = conv_layer(x)
        x = bn_layer(x)
        if (
            time_mlp_layer is not None and t_emb is not None
        ):  # Keep this check for robustness
            time_cond = time_mlp_layer(t_emb)
            x = x + time_cond.unsqueeze(-1).unsqueeze(-1)
        x = relu_layer(x)
        return x

    def forward(self, x_latent_noisy, t):
        x_reshaped = x_latent_noisy.view(
            -1,
            self.latent_reshape_channels,
            self.latent_reshape_side,
            self.latent_reshape_side,
        )
        t_emb = self.time_mlp(t)

        # Initial conv
        s1 = self._apply_integrated_conv_block(
            x_reshaped,
            t_emb,
            self.initial_conv_conv,
            self.initial_conv_bn,
            self.initial_conv_relu,
            self.initial_conv_time_mlp,
        )

        # Down 1
        x_d1 = self._apply_integrated_conv_block(
            s1,
            t_emb,
            self.down1_conv1_conv,
            self.down1_conv1_bn,
            self.down1_conv1_relu,
            self.down1_conv1_time_mlp,
        )
        s2 = self.down1_pool(x_d1)

        # Down 2
        x_d2 = self._apply_integrated_conv_block(
            s2,
            t_emb,
            self.down2_conv1_conv,
            self.down2_conv1_bn,
            self.down2_conv1_relu,
            self.down2_conv1_time_mlp,
        )
        x_bottleneck_in = self.down2_pool(x_d2)

        # Bottleneck
        x_b = self._apply_integrated_conv_block(
            x_bottleneck_in,
            t_emb,
            self.bottleneck_conv1_conv,
            self.bottleneck_conv1_bn,
            self.bottleneck_conv1_relu,
            self.bottleneck_conv1_time_mlp,
        )
        x_b = self._apply_integrated_conv_block(
            x_b,
            t_emb,
            self.bottleneck_conv2_conv,
            self.bottleneck_conv2_bn,
            self.bottleneck_conv2_relu,
            self.bottleneck_conv2_time_mlp,
        )

        # Up 1
        x_u1 = self.up1_deconv(x_b)
        x_u1 = torch.cat([x_u1, s2], dim=1)  # s2 from down1 output (after pooling)
        x_u1 = self._apply_integrated_conv_block(
            x_u1,
            t_emb,
            self.up1_conv1_conv,
            self.up1_conv1_bn,
            self.up1_conv1_relu,
            self.up1_conv1_time_mlp,
        )

        # Up 2
        x_u2 = self.up2_deconv(x_u1)
        x_u2 = torch.cat([x_u2, s1], dim=1)  # s1 from initial_conv output
        x_u2 = self._apply_integrated_conv_block(
            x_u2,
            t_emb,
            self.up2_conv1_conv,
            self.up2_conv1_bn,
            self.up2_conv1_relu,
            self.up2_conv1_time_mlp,
        )

        out_noise_reshaped = self.final_conv(x_u2)
        return out_noise_reshaped.view(-1, self.latent_dim)


class LDM(nn.Module):
    def __init__(
        self,
        vae_model,
        latent_dim=128,
        ldm_unet_latent_reshape_channels=8,
        ldm_unet_time_emb_dim=256,
        ldm_unet_init_channels=64,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    ):
        super().__init__()
        self.vae_encoder = vae_model.encoder
        self.vae_decoder = vae_model.decoder
        self.latent_dim = latent_dim

        self.unet = UNetLatent(
            latent_dim=latent_dim,
            latent_reshape_channels=ldm_unet_latent_reshape_channels,
            time_emb_dim=ldm_unet_time_emb_dim,
            init_unet_channels=ldm_unet_init_channels,
        )

        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.register_buffer("_betas", self.betas)
        self.register_buffer("_alphas", self.alphas)
        self.register_buffer("_alphas_cumprod", self.alphas_cumprod)
        self.register_buffer("_alphas_cumprod_prev", self.alphas_cumprod_prev)
        self.register_buffer("_sqrt_alphas_cumprod", self.sqrt_alphas_cumprod)
        self.register_buffer(
            "_sqrt_one_minus_alphas_cumprod", self.sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("_posterior_variance", self.posterior_variance)

    def _get_tensor_values(self, t, tensor_schedule_name):
        tensor_schedule = getattr(self, f"_{tensor_schedule_name}")
        return tensor_schedule[t].to(t.device).view(-1, 1)

    def forward_diffusion(self, x0_latent, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0_latent)

        sqrt_alphas_cumprod_t = self._get_tensor_values(t, "sqrt_alphas_cumprod")
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(
            t, "sqrt_one_minus_alphas_cumprod"
        )

        xt_latent = (
            sqrt_alphas_cumprod_t * x0_latent + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return xt_latent, noise

    @torch.no_grad()
    def denoise_sample(self, xt_latent, t):
        betas_t = self._get_tensor_values(t, "betas")
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(
            t, "sqrt_one_minus_alphas_cumprod"
        )
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self._get_tensor_values(t, "alphas"))

        predicted_noise = self.unet(xt_latent, t)

        model_mean = sqrt_recip_alphas_t * (
            xt_latent - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0].item() == 0:
            return model_mean
        else:
            posterior_variance_t = self._get_tensor_values(t, "posterior_variance")
            noise = torch.randn_like(xt_latent)  # Fresh noise for this step
            return (
                model_mean + torch.sqrt(posterior_variance_t) * noise
            )  # Add stochasticity

    @torch.no_grad()
    def sample(self, batch_size=1, device="cpu"):
        self.unet.to(device)
        if hasattr(self, "vae_decoder"):
            self.vae_decoder.to(device)

        xt_latent = torch.randn((batch_size, self.latent_dim), device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
            xt_latent = self.denoise_sample(xt_latent, t_tensor)

        sampled_images = self.vae_decoder(xt_latent)
        return sampled_images

    def get_loss(self, x_orig_img, t=None):
        if hasattr(self, "vae_encoder"):
            self.vae_encoder.to(x_orig_img.device)

        with torch.no_grad():
            mu_x0, _ = self.vae_encoder(x_orig_img)
        x0_latent = mu_x0

        if t is None:
            t = torch.randint(
                0, self.num_timesteps, (x_orig_img.shape[0],), device=x_orig_img.device
            ).long()

        xt_latent, true_noise = self.forward_diffusion(x0_latent, t)
        predicted_noise = self.unet(xt_latent, t)
        loss = F.mse_loss(true_noise, predicted_noise)
        return loss

    @torch.no_grad()
    def get_val_reconstruction_metrics(self, x_orig_img, val_t_step=None):
        self.vae_encoder.to(x_orig_img.device)
        self.vae_decoder.to(x_orig_img.device)
        self.unet.to(x_orig_img.device)
        self.unet.eval()

        with torch.no_grad():
            mu_x0, _ = self.vae_encoder(x_orig_img)
        x0_latent = mu_x0

        if val_t_step is None:
            val_t_step = self.num_timesteps // 2

        t = torch.full(
            (x_orig_img.shape[0],), val_t_step, device=x_orig_img.device
        ).long()

        xt_latent, _ = self.forward_diffusion(x0_latent, t)

        predicted_noise_from_xt = self.unet(xt_latent, t)

        sqrt_alphas_cumprod_t = self._get_tensor_values(t, "sqrt_alphas_cumprod")
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(
            t, "sqrt_one_minus_alphas_cumprod"
        )

        x0_pred_latent = (
            xt_latent - sqrt_one_minus_alphas_cumprod_t * predicted_noise_from_xt
        ) / sqrt_alphas_cumprod_t

        reconstructed_img_from_ldm = self.vae_decoder(x0_pred_latent)

        batch_size = x_orig_img.size(0)
        reconstruction_loss = F.binary_cross_entropy(
            reconstructed_img_from_ldm.view(batch_size, -1),
            x_orig_img.view(batch_size, -1),
            reduction="sum",
        )

        return reconstruction_loss / batch_size


if __name__ == "__main__":
    VAE_model = VAE(latent_dim=128)

    total_params = sum(p.numel() for p in VAE_model.parameters())
    encoder_params = sum(p.numel() for p in VAE_encoder().parameters())
    decoder_params = sum(p.numel() for p in VAE_decoder().parameters())
    print(f"Total number of parameters: {total_params}")
    print(f"number of encoder parameters: {encoder_params}")
    print(f"number of decoder parameters: {decoder_params}")
