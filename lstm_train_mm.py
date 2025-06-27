from moe_vae import MoE_VAE, load_mmvae_model
from dataset.m_dataloader import MDataset
import torch
from lstm_model import MuLogvarLSTM
from helpers.utils_proj import device, plot_loss
from tqdm import trange

EPOCHS = 320
BATCH_SIZE = 128
LEARNING_RATE = 0.001
LATENT_DIM = 30

dataset = MDataset("./dataset", "no_obj")
model = load_mmvae_model(f"./models/moe_vae_model_{LATENT_DIM}.pth", 20, LATENT_DIM)
model.to(device)

lstm_model = MuLogvarLSTM(
    embedding_dim=LATENT_DIM,
    hidden_dim=512,
    num_layers=2,
    dropout=0.1
).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

losses = {
    "train_loss": [],
    "val_loss": []
}


random_numbers = [4, 10, 8, 16, 5, 20, 7, 12, 14, 6]
#random_numbers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
for i in random_numbers:
    print(f"Training LSTM model without teacher forcing an len {i}...")
    dataset.prepare_hidden_sequence(model, seq_len=i, device=device)
    tr_loader, vl_loader = dataset.get_sequence_loaders(test_perc=0.3, batch_size=BATCH_SIZE, shuffle=True)
    epoch_tr_loss = 0.0
    epoch_vs_loss = 0.0
    for epoch in trange(EPOCHS, desc=f"SeqLen {i}", unit="epoch", colour="green"):
        epoch_tr_loss = lstm_model.train_epoch(tr_loader, optimizer, device, teacher_forcing=False)
        epoch_vs_loss = lstm_model.test_epoch(vl_loader, device, teacher_forcing=False)
        losses["train_loss"].append(epoch_tr_loss)
        losses["val_loss"].append(epoch_vs_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_tr_loss:.4f} - Val Loss: {epoch_vs_loss:.4f}")
        # early stopping criteria
        if epoch > 10 and epoch_vs_loss > max(losses["val_loss"][-6:-1]):
            print("\nEarly stopping triggered.\n")
            lstm_model.load_state_dict(torch.load(f"./models/lstm_mmvae_{LATENT_DIM}.pth", map_location=device))
            break
        # Save model if validation loss improves
        if epoch == 0 or epoch_vs_loss < min(losses["val_loss"][:-1]):
            torch.save(lstm_model.state_dict(), f"./models/lstm_mmvae_{LATENT_DIM}.pth")
            print(f"Model saved for sequence length {i}.")

# Plot the losses
plot_loss(losses, "LSTM Training Losses", "Epochs", "Loss",
          save_path="./models/lstm_training_losses_mm.png")


