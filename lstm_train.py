import torch
from vae import VAE
from lstm_model import MuLogvarLSTM
from dataset.ordered_dataloader import Dataset
from utils_proj import device


LATENT_DIM = 200
VAE_NAME = "vae_model_final.pth"
"models/vae_final_model.pth"
def main():

	model = VAE(latent_dim=LATENT_DIM).to(device)
	model.load_state_dict(torch.load("models/vae_final_model.pth", map_location=device))
	ds = Dataset("dataset", "no_obj", model)
	tr, vs = ds.get_training_set(), ds.get_validation_set()
	lstm_model = MuLogvarLSTM(embedding_dim=LATENT_DIM, hidden_dim=256, num_layers=2, dropout=0.2).to(device)

	print(f"Training set length: {len(tr)}")
	print(f"Validation set length: {len(vs)}")
	print(f"Model: {lstm_model}")

	lstm_model.train()
	tr_loader = torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True)
	for mu, logvar, act in tr_loader:
		mu, logvar, act = mu.to(device), logvar.to(device), act.to(device)
		lstm_model.zero_grad()
		loss = lstm_model.forward_training(mu, logvar, act)


if __name__ == "__main__":
	main()