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
    """A convolutional VAE tailored for 28×28 grayscale images."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        # -------- Encoder --------
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 28 → 14
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14 → 7
        self.enc_fc = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # -------- Decoder --------
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 64 * 7 * 7)
        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 7 → 14
        self.dec_deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 14 → 28

    # ---- Helper sub‑functions ----
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.view(x.size(0), -1)
        h = F.relu(self.enc_fc(h))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.dec_deconv1(h))
        return torch.sigmoid(self.dec_deconv2(h))

    # ---- Forward pass ----
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ------------- Loss -------------

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Binary cross‑entropy + KL divergence, averaged per mini‑batch."""
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


# ---------- Train / Test loops ----------

def train_epoch(model: VAE, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    epoch_loss = 0.0
    for x, _ in loader:
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
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            test_loss += loss.item() * x.size(0)
    return test_loss / len(loader.dataset)


# ---------- Utility ----------

def save_sample_grid(model: VAE, epoch: int, device: torch.device, out_dir: Path) -> None:
    """Save a 4×4 grid of prior samples and reconstructions to disk for quick visual feedback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        # ----- Prior samples -----
        z = torch.randn(16, model.fc_mu.out_features, device=device)
        samples = model.decode(z).cpu()
        utils.save_image(samples, out_dir / f"samples_epoch{epoch:03d}.png", nrow=4)


# ------------- Main script -------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Variational Auto‑Encoder on MNIST")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    # Determinism for reproducibility
    torch.manual_seed(42)

    # Data
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model & Optimizer
    model = VAE(args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = test_epoch(model, test_loader, device)
        print(f"Epoch {epoch:02d}: Train loss = {train_loss:.4f} | Test loss = {test_loss:.4f}")
        save_sample_grid(model, epoch, device, args.out_dir)

    # Save final model
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "checkpoints" / "vae_mnist_final.pt")


if __name__ == "__main__":
    main()
