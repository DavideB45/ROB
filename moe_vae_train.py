from moe_vae import MoE_VAE, create_mmvae_model, save_mmvae_model, load_mmvae_model
from dataset.m_dataloader import MDataset
from helpers.utils_proj import get_best_device, show_image, plot_loss
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

	num_epochs = 100
	losses = {'train_loss': [], 'val_loss': []}
	train_loss = 0.0
	val_loss = 0.0
	for epoch in tqdm.trange(num_epochs, desc="Epochs", unit="epoch", postfix={'train_loss': train_loss, 'val_loss': val_loss}):
		train_loss = model.train_epoch(train_loader, ['VISION', 'POS'], optimizer, beta=BETA, device=device)
		val_loss = model.test_epoch(val_loader, ['VISION', 'POS'], beta=BETA, device=device)
		print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
		losses['train_loss'].append(train_loss)
		losses['val_loss'].append(val_loss)
		# define early stopping criteria
		if epoch > 10 and val_loss > max(losses['val_loss'][-10:-1]):
			print("Early stopping triggered.")
			break
		# save model if validation loss improves
		if epoch == 0 or val_loss < min(losses['val_loss'][:-1]):
			save_mmvae_model(model, "moe_vae_model.pth")
			print(f'Model saved at epoch {epoch+1} with validation loss {val_loss:.4f}')
	# Plot the training and validation loss
	plot_loss(losses, title='MoE VAE Training Loss',
			  xlabel='Epochs', ylabel='Loss', save_path='moe_vae_training_loss.png')

def try_use_model(name: str):
	'''
	Try to use the model with the given name.
	'''
	try:
		model = load_mmvae_model(name, 20, LATENT_DIM)
		print(f'Model {name} loaded successfully.')
		model.to(device)
		_, loader = get_dataset()
		input_data = next(iter(loader))
		input_data = [x.to(device) for x in input_data]
		with torch.no_grad():
			vision_output, proprioception_output, mu, logvar = model.forward(input_data[0], input_data[1])
			for i in range(min(len(vision_output), 5)):
				show_image(vision_output[i].cpu())
	except FileNotFoundError:
		print(f'Model {name} not found.')
		return None
	
if __name__ == "__main__":
	main()
	try_use_model("moe_vae_model.pth")