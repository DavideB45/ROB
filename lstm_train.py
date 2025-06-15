import torch
from vae import VAE
from lstm_model import MuLogvarLSTM
from dataset.ordered_dataloader import Dataset
from helpers.utils_proj import device
import random
import matplotlib.pyplot as plt

LATENT_DIM = 200
VAE_NAME = "vae_model_foundation_kl04_l2e4.pth"
"vae_model.pth"
"models/vae_final_model.pth"
def main():

	model = VAE(latent_dim=LATENT_DIM).to(device)
	model.load_state_dict(torch.load(VAE_NAME, map_location=device))
	lstm_model = MuLogvarLSTM(embedding_dim=LATENT_DIM, hidden_dim=512, num_layers=2, dropout=0.1).to(device)
	print(f"Model: {lstm_model}")

	losses = {
		"train": [],
		"validation": []
	}
	lstm_model.train()
	
	random_numbers = [random.randint(4, 20) for _ in range(10)]
	for i in random_numbers:
		print(f"Training LSTM model without teacher forcing an len {i}...")
		ds = Dataset("dataset", "no_obj", model, seq_len=i)
		tr, vs = ds.get_training_set(), ds.get_validation_set()
		tr_loader = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)
		vs_loader = torch.utils.data.DataLoader(vs, batch_size=32, shuffle=False)
		optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0001, weight_decay=0.0001)
		for epoch in range(320):
			epoch_tr_loss = lstm_model.train_epoch(tr_loader, optimizer, device, teacher_forcing=False)
			epoch_vs_loss = lstm_model.test_epoch(vs_loader, device, teacher_forcing=False)
			print(f"Epoch {epoch+1}: Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_vs_loss:.4f}")
			losses["train"].append(epoch_tr_loss)
			losses["validation"].append(epoch_vs_loss)
			if epoch > 20 and epoch_vs_loss > max(losses["validation"][-6:-1]):
				print("Early stopping triggered.")
				break

	for i in []:
		print(f"Training LSTM model without teacher forcing an len {i}...")
		ds = Dataset("dataset", "no_obj", model, seq_len=i)
		tr, vs = ds.get_training_set(), ds.get_validation_set()
		tr_loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
		vs_loader = torch.utils.data.DataLoader(vs, batch_size=32, shuffle=False)
		optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.00001, weight_decay=0.0001)
		for epoch in range(120):
			epoch_tr_loss = lstm_model.train_epoch(tr_loader, optimizer, device, teacher_forcing=False, full_error=True)
			epoch_vs_loss = lstm_model.test_epoch(vs_loader, device, teacher_forcing=False, full_error=True)
			print(f"Epoch {epoch+1}: Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_vs_loss:.4f}")
			losses["train"].append(epoch_tr_loss)
			losses["validation"].append(epoch_vs_loss)

	# ds = Dataset("dataset", "no_obj", model, seq_len=20)
	# tr, vs = ds.get_training_set(), ds.get_validation_set()
	# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.00004, weight_decay=0e-3)
	# tr_loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
	# vs_loader = torch.utils.data.DataLoader(vs, batch_size=32, shuffle=False)
	# print("Traoning the LSTM model with teacher forcing...")
	# for epoch in range(200):
	# 	epoch_tr_loss = lstm_model.train_epoch(tr_loader, optimizer, device, teacher_forcing=True)
	# 	epoch_vs_loss = lstm_model.test_epoch(vs_loader, device, teacher_forcing=True)
	# 	print(f"Epoch {epoch+1}: Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_vs_loss:.4f}")
	# 	losses["train"].append(epoch_tr_loss)
	# 	losses["validation"].append(epoch_vs_loss)


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