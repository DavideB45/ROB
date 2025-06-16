from fc_vae import FC_VAE
from vae_model import VAE
import torch
import torch.nn as nn


class MoE_VAE(nn.Module):
	def __init__(self, vision: VAE, proprioception: FC_VAE, latent_dim: int = 20):
		super(MoE_VAE, self).__init__()
		self.vision = vision
		self.proprioception = proprioception
		self.latent_dim = latent_dim

	def encode(self, vision_input, proprioception_input):
		# Encode vision input
		vision_mu, vision_logvar = self.vision.encode(vision_input)
		mu, logvar = self.proprioception.encode(proprioception_input)
		# Combine the latent representations
		combined_mu = (vision_mu + mu) / 2
		combined_logvar = (vision_logvar + logvar) / 2
		return combined_mu, combined_logvar
	
	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		# Reparameterization trick
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std
	
	def decode(self, z):
		# Decode the combined latent representation
		vision_output = self.vision.decode(z)
		proprioception_output = self.proprioception.decode(z)
		return vision_output, proprioception_output
	
	def forward(self, vision_input, proprioception_input):
		# Forward pass through the MoE VAE
		mu, logvar = self.encode(vision_input, proprioception_input)
		z = self.reparameterize(mu, logvar)
		vision_output, proprioception_output = self.decode(z)
		return vision_output, proprioception_output, mu, logvar
	
	def loss_function(self, vision_input, proprioception_input, beta=1.0):
		# Compute the loss function for the MoE VAE
		vision_output, proprioception_output, mu, logvar = self.forward(vision_input, proprioception_input)

		# Reconstruction loss
		# TODO: Adjust loss for vision
		reconstruction_loss_vision = nn.functional.mse_loss(vision_output, vision_input, reduction='sum')
		reconstruction_loss_proprioception = nn.functional.mse_loss(proprioception_output, proprioception_input, reduction='sum')

		# KL divergence loss
		kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta

		return reconstruction_loss_vision + reconstruction_loss_proprioception + kld_loss
	
	def train_epoch(self, vision_loader, proprioception_loader, optimizer, beta=1.0):
		self.train()
		total_loss = 0.0
		for vision_input, proprioception_input in zip(vision_loader, proprioception_loader):
			optimizer.zero_grad()
			loss = self.loss_function(vision_input, proprioception_input, beta)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		return total_loss / len(vision_loader)
	
	def test_epoch(self, vision_loader, proprioception_loader, beta=1.0):
		self.eval()
		total_loss = 0.0
		with torch.no_grad():
			for vision_input, proprioception_input in zip(vision_loader, proprioception_loader):
				loss = self.loss_function(vision_input, proprioception_input, beta)
				total_loss += loss.item()
		return total_loss / len(vision_loader)
	
def create_mmvae_model(proprioception_input_dim: int, latent_dim: int):
	"""
	Create a multimodal VAE model with vision and proprioception branches.
	Args:
		proprioception_input_dim (int): Dimension of the proprioception input.
		latent_dim (int): Dimension of the latent space.
	Returns:
		MoE_VAE: An instance of the MoE_VAE model.
	"""
	vision_model = VAE(latent_dim=latent_dim)
	proprioception_model = FC_VAE(input_dim=proprioception_input_dim, latent_dim=latent_dim)
	return MoE_VAE(vision=vision_model, proprioception=proprioception_model, latent_dim=latent_dim)
	

