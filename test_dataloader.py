from dataset.dataloader import Dataset
import torch
from vae import VAE, vae_loss, train_epoch, test_epoch
import random
import matplotlib.pyplot as plt

dataset = Dataset(
    path="dataset",
    condition="no_obj"
)

def show_datasets():
    train_set = dataset.get_training_set()
    val_set = dataset.get_validation_set()
    print(f"Training set size: {len(train_set[0])}")
    print(f"Validation set size: {len(val_set[0])}")
    sample = dataset[0]
    print(f"Sample data shape: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")

def train_model():
    # train VAE model
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device=device)
    model = VAE(latent_dim=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    num_epochs = 10
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = test_epoch(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
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
    torch.save(model.state_dict(), "vae_model.pth")

def test_model():
    # Load the model
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device=device)
    model = VAE(latent_dim=100).to(device)
    model.load_state_dict(torch.load("vae_model.pth"))
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
    #train_model()
    test_model()