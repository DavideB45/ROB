import torch
from fc_vae import FC_VAE
from dataset.m_dataloader import MDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from helpers.utils_proj import get_best_device
import time

device = get_best_device()
device = torch.device('cpu')
EMBEDDING_DIM = 200  # Dimension of the input data
BETA = 0.1  # Weight for the KL divergence term in the loss function
LEARNING_RATE = 0.0005  # Learning rate for the optimizer
BATCH_SIZE = 64  # Batch size for training

def get_dataset() -> tuple[DataLoader, DataLoader]:
    '''Get the dataset for training and validation.'''
    path = "./dataset"
    scenario = "no_obj"
    dataset = MDataset(path, scenario)
    tr, vl = dataset.get_VAE_loaders(["POS"], test_perc=0.1, batch_size=BATCH_SIZE, shuffle=True)
    return tr, vl

def get_model() -> FC_VAE:
    '''Get the model for training.'''
    input_dim = 20
    model = FC_VAE(input_dim=input_dim, latent_dim=EMBEDDING_DIM)
    model.to(device)
    print(f'Model: {model}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model

def get_optimizer(model: FC_VAE) -> torch.optim.Optimizer:
    '''Get the optimizer for training.'''
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return optimizer

def train(model: FC_VAE, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int = 100) -> dict:
    '''Train the model.'''
    history = {'train_loss': [], 'val_loss': []}
    model.train()
    for epoch in range(epochs):
        total_tr_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0].to(torch.float32).to(device)
            recon_x, mu, logvar = model(x)
            loss = model.loss_function(recon_x, x, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()
            total_tr_loss += loss.item()
        avg_tr_loss = total_tr_loss / len(train_loader)
        history['train_loss'].append(avg_tr_loss)
        
        model.eval()
        total_vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(torch.float32).to(device)
                recon_x, mu, logvar = model(x)
                loss = model.loss_function(recon_x, x, mu, logvar, beta=BETA)
                total_vl_loss += loss.item()
        avg_vl_loss = total_vl_loss / len(val_loader)
        history['val_loss'].append(avg_vl_loss)
        
        print(f'Epoch {(epoch+1):3d}/{epochs}, Train Loss: {avg_tr_loss:.4f}, Val Loss: {avg_vl_loss:.4f}', end='\r')
    print() # Newline after training is complete
    return history

def plot_loss(history: dict):
    '''Plot the training and validation loss.'''
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main():
    '''Main function to run the training.'''
    train_loader, val_loader = get_dataset()
    model = get_model()
    optimizer = get_optimizer(model)
    
    
    start_time = time.time()
    history = train(model, train_loader, val_loader, optimizer, epochs=100)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    plot_loss(history)

    #print some example outputs
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(torch.float32).to(device)
            recon_x, mu, logvar = model(x)
            print(f"Input: {x[0][:5]}")
            print(f"Reconstructed: {recon_x[0][:5]}")
            print(f"Mu: {mu[0][:5]}")
            print(f"LogVar: {logvar[0][:5]}")
            break

if __name__ == "__main__":
    main()