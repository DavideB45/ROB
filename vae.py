"""
Variational Auto‑Encoder (VAE) for MNIST in PyTorch
--------------------------------------------------
Run this file directly to train a convolutional VAE on MNIST.  The script
handles downloading the dataset, training, evaluation and basic sample
generation.

Usage (defaults in brackets):
    python vae_mnist.py --epochs 20 --batch-size 128 --latent-dim 20 --lr 1e-3

Add --no-cuda to force CPU even if a GPU is available.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils



class VAE(nn.Module):
    """A convolutional VAE tailored for 64x64 colored images."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # -------- Encoder --------
        # Input: Batch x 3 x 64 x 64
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # Output: Batch x 32 x 32 x 32
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: Batch x 64 x 16 x 16
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: Batch x 128 x 8 x 8
        
        self.flattened_size = 128 * 8 * 8  # 8192
        self.fc_hidden_dim = 512 # Intermediate dimension for FC layers

        self.enc_fc = nn.Linear(self.flattened_size, self.fc_hidden_dim)
        self.fc_mu = nn.Linear(self.fc_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_hidden_dim, latent_dim)

        # -------- Decoder --------
        self.dec_fc1 = nn.Linear(latent_dim, self.fc_hidden_dim)
        self.dec_fc2 = nn.Linear(self.fc_hidden_dim, self.flattened_size)
        
        # Input: Batch x 128 x 8 x 8 (after reshape)
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: Batch x 64 x 16 x 16
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: Batch x 32 x 32 x 32
        self.dec_deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)   # Output: Batch x 3 x 64 x 64

    # ---- Helper sub‑functions ----
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(x.size(0), -1)  # Flatten
        h = F.relu(self.enc_fc(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 128, 8, 8)  # Reshape to match deconv input
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        return torch.sigmoid(self.dec_deconv3(h)) # Sigmoid for [0,1] output

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    # ---- Forward pass ----
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar



# ------------- Loss -------------

import torch

def create_purple_weight_map_rgb(x: torch.Tensor,
                                 purple_weight: float = 10.0,
                                 dominance_margin: float = 0.1) -> torch.Tensor:
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

    # Unpack the RGB channels
    # x shape is (Batch, Channels, Height, Width)
    red_channel = x[:, 0, :, :]
    green_channel = x[:, 1, :, :]
    blue_channel = x[:, 2, :, :]

    # The rule for being "purple": Red and Blue are dominant over Green.
    # The dominance_margin makes the rule stricter.
    # The mask will be a boolean tensor of shape (B, H, W).
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
    # Ensure the mask is on the same device as x

    # Create the weight map, starting with a base weight of 1.0 for all pixels.
    weights = torch.ones_like(is_purple_mask, dtype=torch.float32, device=x.device)

    # Apply the higher weight to the pixels that match our "purple" rule.
    weights[is_purple_mask] = purple_weight

    # Add a channel dimension for broadcasting against the loss tensor.
    # The shape becomes (B, 1, H, W), which can be multiplied with (B, 3, H, W).
    return weights.unsqueeze(1)

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, purple_weight: float = 5.0, dominance_margin: float = 0.1) -> torch.Tensor:
    """Binary cross‑entropy + KL divergence, averaged per mini‑batch."""
    weight_in_map = create_purple_weight_map_rgb(x, purple_weight, dominance_margin)
    weight_out_map = create_purple_weight_map_rgb(recon_x, purple_weight, dominance_margin)
    bce = F.binary_cross_entropy(recon_x, x, reduction="none")
    # Apply the weight map to the BCE loss
    bce = (bce * (weight_in_map + weight_out_map)).sum()
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + 0.8*kld) / x.size(0)


# ---------- Train / Test loops ----------

def train_epoch(model: VAE, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    epoch_loss = 0.0
    for x, _, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
    return epoch_loss / len(loader.dataset)


def test_epoch(model: VAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            test_loss += loss.item() * x.size(0)
    return test_loss / len(loader.dataset)