from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from helpers.resBlock import ResidualBlock
from tqdm import tqdm



class VAE(nn.Module):
    """A convolutional VAE tailored for 64x64 colored images."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            ResidualBlock(3, 32, downsample=True),   # 64x64 → 32x32
            ResidualBlock(32, 64, downsample=True),  # 32x32 → 16x16
            ResidualBlock(64, 128, downsample=True), # 16x16 → 8x8
            ResidualBlock(128, 256, downsample=True) # 8x8 → 4x4
        )
        self.enc_fc = nn.Linear(256 * 4 * 4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # ---------- Decoder ----------
        self.dec_fc = nn.Linear(latent_dim, 512)
        self.dec_expand = nn.Linear(512, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 → 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32x32 → 64x64
        )


    # ---- Helper sub‑functions ----
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        h = F.leaky_relu(self.enc_fc(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = F.leaky_relu(self.dec_fc(z))
        h = F.leaky_relu(self.dec_expand(h)).view(-1, 256, 4, 4)
        return self.decoder(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar



# ------------- Loss -------------

def create_purple_weight_map_rgb(x: torch.Tensor,
                                 purple_weight: float = 2.0) -> torch.Tensor:
    """
    Creates a weight map that emphasizes purple pixels, working directly in RGB.

    Args:
        x (torch.Tensor): Input batch of images in (B, 3, H, W) format.
                          Assumes RGB order and values normalized to [0, 1].
        purple_weight (float): The weight to apply to pixels identified as purple.
        dominance_margin (float): How much larger R and B must be than G to be
                                  considered purple. Helps avoid grayish colors.

    Returns:
        torch.Tensor: A weight map of shape (B, 1, H, W).
    """
    # Ensure input is in the right format (B, 3, H, W)
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"Input tensor must be in (B, 3, H, W) format, but got {x.shape}")

    # x shape is (Batch, Channels, Height, Width)
    red_channel = x[:, 0, :, :]
    green_channel = x[:, 1, :, :]
    blue_channel = x[:, 2, :, :]
    is_purple_mask = (red_channel > 70/255) & \
                     (green_channel > 60/255) & \
                     (blue_channel > 100/255) & \
                     (red_channel < 127/255) & \
                     (green_channel < 100/255) & \
                     (blue_channel < 160/255)
    
    # import matplotlib.pyplot as plt
    # Debugging: Visualize the mask
    # plt.imshow(is_purple_mask[0].cpu().numpy(), cmap='gray')
    # plt.title("Purple Mask Visualization")
    # plt.axis('off')
    # plt.show()

    weights = torch.ones_like(is_purple_mask, dtype=torch.float32, device=x.device)
    weights[is_purple_mask] = purple_weight
    # Add a channel dimension for broadcasting against the loss tensor.
    # The shape becomes (B, 1, H, W), which can be multiplied with (B, 3, H, W).
    return weights.unsqueeze(1)

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor, kl_weight:float=1.0,
            perceptual_loss_fn = None, perc_weight:float=1.0,
            purple_weight:float = 2) -> torch.Tensor:
    """Binary cross entropy + KL divergence, averaged per mini batch."""
    tot_loss = 0.0
    if perceptual_loss_fn is not None:
        perc_loss = perceptual_loss_fn(recon_x, x)
        tot_loss += perc_weight * perc_loss
    if purple_weight > 0:
        bce = F.mse_loss(recon_x, x, reduction="none")
        weight_in_map = create_purple_weight_map_rgb(x, purple_weight)
        weight_out_map = create_purple_weight_map_rgb(recon_x, purple_weight)
        bce = (bce * (weight_in_map + weight_out_map)/2).sum()
        tot_loss += bce / x.size(0)  # Average over the batch
    else:
        tot_loss += F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    tot_loss += kl_weight * kld
    return tot_loss

def compute_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between the learned distribution and a standard normal distribution."""
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


# ---------- Train / Test loops ----------

def train_epoch(model: VAE,
                loader: DataLoader, 
                optimizer: optim.Optimizer, 
                device: torch.device, 
                kl_weight:float=1.0,
                perceptual_loss_fn = None,
                perc_weight:float=1.0,
                purple_weight:float = 2.0
                ) -> float:
    model.train()
    epoch_loss = 0.0
    for x in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, 
                        mu, logvar, kl_weight,
                        perceptual_loss_fn, perc_weight,
                        purple_weight=purple_weight
                        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
        del x
        torch.cuda.empty_cache()
    return epoch_loss / len(loader.dataset)


def test_epoch(model: VAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            test_loss += loss.item() * x.size(0)
            del x
            torch.cuda.empty_cache()
    return test_loss / len(loader.dataset)