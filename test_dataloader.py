from dataset.dataloader import Dataset
import torch
from vae import VAE, vae_loss, train_epoch, test_epoch
import random
import matplotlib.pyplot as plt
from helpers.utils_proj import get_best_device, show_datasets

dataset:Dataset = Dataset(
    path="dataset",
    condition="no_obj"
)
device = get_best_device()


def train_model():
    model = VAE(latent_dim=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = torch.utils.data.DataLoader(dataset.get_training_set()[0], batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset.get_validation_set()[0], batch_size=64, shuffle=False)
    num_epochs = 40
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        kl_weight = min(1.0, epoch / (num_epochs * 0.7))  # gradually increase KL weight
        train_loss = train_epoch(model, train_loader, optimizer, device, kl_weight=0.1)
        test_loss = test_epoch(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        # early stopping condition
        if epoch > 5 and test_loss > max(test_losses[-6:]):
            print("Early stopping triggered.")
            break
        # save the best model
        if epoch == 0 or test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), "vae_model.pth")
    # Plot training and test losses
    plt.figure(figsize=(10, 5))
    plt.title("Training and Test Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # Save the model
    #torch.save(model.state_dict(), "vae_model.pth")

def test_model(name: str = "vae_model.pth"):
    model = VAE(latent_dim=200).to(device)
    model.load_state_dict(torch.load(name))
    model.eval()
    
    # Test the model with a couple of samples
    num_samples_to_show = 2
    fig, axes = plt.subplots(num_samples_to_show, 2, figsize=(8, num_samples_to_show * 4))
    fig.suptitle("Original vs. Reconstructed Samples", fontsize=16)

    for i in range(num_samples_to_show):
        # Get a sample
        idx = random.randint(0, len(dataset) - 1)
        original_sample_img = dataset[idx][0] # Assuming the image is the first element
        sample_to_model = original_sample_img.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            recon, mu, logvar = model(sample_to_model)

        # Prepare original for display
        original_display = original_sample_img.permute(1, 2, 0).cpu().numpy() # HWC
        if original_display.shape[2] == 1: # Grayscale
            original_display = original_display.squeeze(2)

        # Prepare reconstruction for display
        recon_display = recon.squeeze(0)  # Remove batch dimension
        recon_display = recon_display.permute(1, 2, 0).cpu().numpy() # HWC
        if recon_display.shape[2] == 1: # Grayscale
            recon_display = recon_display.squeeze(2)
        
        # Display original
        ax = axes[i, 0]
        ax.imshow(original_display, cmap='gray' if original_display.ndim == 2 else None)
        ax.set_title(f"Original Sample {i+1}")
        ax.axis('off')

        # Display reconstruction
        ax = axes[i, 1]
        ax.imshow(recon_display, cmap='gray' if recon_display.ndim == 2 else None)
        ax.set_title(f"Reconstructed Sample {i+1}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()


if __name__ == "__main__":
    train_model()
    #show_datasets()
    test_model("vae_model.pth")