from moe_vae import MoE_VAE, create_mmvae_model, save_mmvae_model
from dataset.m_dataloader import MDataset
from helpers.utils_proj import get_best_device
import torch
import tqdm

LATENT_DIM = 200  # Dimension of the latent space
BETA = 0.1  # Weight for the KL divergence term in the loss function
LEARNING_RATE = 0.001  # Learning rate for the optimizer
BATCH_SIZE = 64  # Batch size for training
device = get_best_device()

def get_dataset() -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
	'''
	Get the dataset for training and validation.
	Containing VISION and POS
	'''
	path = "./dataset"
	scenario = "no_obj"
	dataset = MDataset(path, scenario)
	train_loader, val_loader = dataset.get_VAE_loaders(["VISION", "POS"],
													   test_perc=0.1,
													   batch_size=32,
													   shuffle=True)
	return train_loader, val_loader

def get_model() -> MoE_VAE:
	'''Get the model for training.'''
	return create_mmvae_model(
		proprioception_input_dim=20,
		latent_dim=LATENT_DIM
	)
	

def main():
	'''Main function to train the MoE_VAE model.'''
	train_loader, val_loader = get_dataset()
	model = get_model()
	model.to(device)
	print(f'Model: {model}')
	print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	num_epochs = 40
	for epoch in tqdm.trange(num_epochs, desc="Epochs"):
		train_loss = model.train_epoch(train_loader, ['VISION', 'POS'], optimizer, beta=BETA, device=device)
		val_loss = model.test_epoch(val_loader, ['VISION', 'POS'], beta=BETA, device=device)
		print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
	# Save the model after training
	save_mmvae_model(model, "moe_vae_model.pth")
if __name__ == "__main__":
	main()