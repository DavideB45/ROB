import torch
from vae import VAE
from lstm_model import MuLogvarLSTM
from dataset.ordered_dataloader import Dataset
from utils_proj import device
import matplotlib.pyplot as plt

LATENT_DIM = 200
VAE_NAME = "vae_model_final.pth"
"models/vae_final_model.pth"
def main():

	model = VAE(latent_dim=LATENT_DIM).to(device)
	model.load_state_dict(torch.load("vae_model.pth", map_location=device))
	ds = Dataset("dataset", "no_obj", model, seq_len=20)
	tr, vs = ds.get_training_set(), ds.get_validation_set()
	lstm_model = MuLogvarLSTM(embedding_dim=LATENT_DIM, hidden_dim=256, num_layers=1, dropout=0).to(device)

	print(f"Training set length: {len(tr)}")
	print(f"Validation set length: {len(vs)}")
	print(f"Model: {lstm_model}")

	lstm_model.train()
	optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0002, weight_decay=0e-3)
	tr_loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
	vs_loader = torch.utils.data.DataLoader(vs, batch_size=32, shuffle=False)
	losses = {
		"train": [],
		"validation": []
	}
	print("Traoning the LSTM model with teacher forcing...")
	for epoch in range(10):
		epoch_tr_loss = lstm_model.train_epoch(tr_loader, optimizer, device, teacher_forcing=True)
		epoch_vs_loss = lstm_model.test_epoch(vs_loader, device, teacher_forcing=True)
		print(f"Epoch {epoch+1}: Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_vs_loss:.4f}")
		losses["train"].append(epoch_tr_loss)
		losses["validation"].append(epoch_vs_loss)

	for i in range(5):
		print(f"Training LSTM model without teacher forcing an len {(i + 1)*15}...")
		ds = Dataset("dataset", "no_obj", model, seq_len=(i + 1)*15)
		tr, vs = ds.get_training_set(), ds.get_validation_set()
		optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0001, weight_decay=0e-3)
		for epoch in range(100):
			epoch_tr_loss = lstm_model.train_epoch(tr_loader, optimizer, device, teacher_forcing=False)
			epoch_vs_loss = lstm_model.test_epoch(vs_loader, device, teacher_forcing=False)
			print(f"Epoch {epoch+1}: Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_vs_loss:.4f}")
			losses["train"].append(epoch_tr_loss)
			losses["validation"].append(epoch_vs_loss)

	

	# Plotting the losses
	plt.figure(figsize=(10, 5))
	plt.plot(losses["train"], label="Train Loss")
	plt.plot(losses["validation"], label="Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training and Validation Losses")
	plt.legend()
	plt.show()

	# Save the trained model
	torch.save(lstm_model.state_dict(), "models/lstm_final_model.pth")

if __name__ == "__main__":
	main()