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
	
	def position_to_vision(self, proprioception_input):
		"""
		Convert proprioception input to vision output.
		This method assumes that the proprioception model can generate a latent representation
		that can be decoded into a vision-like output.
		"""
		mu, logvar = self.proprioception.encode(proprioception_input)
		z = self.reparameterize(mu, logvar)
		return self.vision.decode(z)
	
	def pos_to_vis_loss(self, pos_input, vision_output, beta=1.0):
		"""
		Calculate the loss for converting position to vision.
		This method assumes that the proprioception model can generate a latent representation
		that can be decoded into a vision-like output.
		"""
		pos_output = self.position_to_vision(pos_input)
		reconstruction_loss = nn.functional.mse_loss(vision_output, pos_output, reduction='sum') / pos_input.size(0)
		kld_loss = self.kl_divergence_loss(*self.proprioception.encode(pos_input)) * beta
		return reconstruction_loss + kld_loss
	
	def visual_loss(self, vision_input, vision_output):
		"""
		Calculate the loss for the vision output.
		TODO: This should be customized based on the specific loss function used for vision data.
		"""
		return nn.functional.mse_loss(vision_output, vision_input, reduction='sum') / vision_input.size(0)
	
	def proprioception_loss(self, proprioception_input, proprioception_output):
		"""
		Calculate the loss for the proprioception output.
		"""
		return nn.functional.mse_loss(proprioception_output, proprioception_input, reduction='sum') / proprioception_input.size(0)
	
	def kl_divergence_loss(self, mu, logvar):
		"""
		Calculate the KL divergence loss.
		"""
		return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	
	def loss_function(self, vision_input, proprioception_input, beta=1.0):
		vision_output, proprioception_output, mu, logvar = self.forward(vision_input, proprioception_input)
		# TODO: Adjust loss for vision
		reconstruction_loss_vision = nn.functional.mse_loss(vision_output, vision_input, reduction='sum') / vision_input.size(0)
		reconstruction_loss_proprioception = nn.functional.mse_loss(proprioception_output, proprioception_input, reduction='sum') / proprioception_input.size(0)
		kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * beta

		return reconstruction_loss_vision + reconstruction_loss_proprioception + kld_loss

	def train_epoch(self, dataloader, sense_order:list[str], optimizer, beta=1.0, device='cpu'):
		self.train()
		total_loss = 0.0
		if len(sense_order) != 2:
			raise ValueError("sense_order must contain exactly two modalities: ['VISION', 'POS']")
		if 'VISION' not in sense_order or 'POS' not in sense_order:
			raise ValueError("sense_order must contain 'VISION' and 'POS'")

		vision_index = sense_order.index('VISION')
		pos_index = sense_order.index('POS')
		for inputs in dataloader:
			optimizer.zero_grad()
			loss = self.loss_function(
							inputs[vision_index].to(device), 
							inputs[pos_index].to(device),
							beta)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		return total_loss / len(dataloader)
	
	def test_epoch(self, dataloader, sense_order:list[str], beta=1.0, device='cpu'):
		self.eval()
		total_loss = 0.0
		if len(sense_order) != 2:
			raise ValueError("sense_order must contain exactly two modalities: ['VISION', 'POS']")
		if 'VISION' not in sense_order or 'POS' not in sense_order:
			raise ValueError("sense_order must contain 'VISION' and 'POS'")
		
		vision_index = sense_order.index('VISION')
		pos_index = sense_order.index('POS')
		with torch.no_grad():
			for inputs in dataloader:
				loss = self.loss_function(
								inputs[vision_index].to(device), 
								inputs[pos_index].to(device),
								beta)
				total_loss += loss.item()
		return total_loss / len(dataloader)
	
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

def save_mmvae_model(model: MoE_VAE, path: str):
	"""
	Save the multimodal VAE model to a file.
	Args:
		model (MoE_VAE): The multimodal VAE model to save.
		path (str): The file path to save the model.
	"""
	torch.save(model.state_dict(), path)

def load_mmvae_model(path: str, proprioception_input_dim: int, latent_dim: int) -> MoE_VAE:
	"""
	Load a multimodal VAE model from a file.
	Args:
		path (str): The file path to load the model from.
		proprioception_input_dim (int): Dimension of the proprioception input.
		latent_dim (int): Dimension of the latent space.
	Returns:
		MoE_VAE: An instance of the MoE_VAE model loaded from the file.
	"""
	model = create_mmvae_model(proprioception_input_dim, latent_dim)
	model.load_state_dict(torch.load(path))
	return model
	

