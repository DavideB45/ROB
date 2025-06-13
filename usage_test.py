from vae import VAE
from lstm_model import MuLogvarLSTM
from dataset.ordered_dataloader import Dataset
from utils_proj import get_best_device, image_list_to_gif
import torch

if __name__ == "__main__":

    # Set the device to CUDA if available, otherwise CPU
    device = get_best_device()

    # Initialize the VAE model
    latent_dim = 200
    vae_model = VAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load("models/vae_final_model.pth", map_location=device))
    vae_model.eval()  # Set the model to evaluation mode

    # Create the dataset
    dataset = Dataset("dataset", "no_obj", vae_model, seq_len=24)

    # Get training and validation sets
    train_set, val_set = dataset.get_training_set(), dataset.get_validation_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)


    # Initialize the LSTM model
    lstm_model = MuLogvarLSTM(embedding_dim=latent_dim, hidden_dim=512, num_layers=2, dropout=0.1).to(device)
    lstm_model.load_state_dict(torch.load("models/lstm_final_model.pth", map_location=device))
    lstm_model.eval()  # Set the model to evaluation mode

    print("Running inference with teacher forcing...")
    # take a sample from the dataset
    sample_mu, sample_logvar, sample_act = train_loader.__iter__().__next__()
    sample_mu = sample_mu.float().to(device)
    sample_logvar = sample_logvar.float().to(device)
    sample_act = sample_act.float().to(device)
    mu, logvar = lstm_model.predict(sample_mu, sample_logvar, sample_act)

    for i in range(len(mu)):
        mu[i] = dataset.rescale(mu[i], "MU")
        logvar[i] = dataset.rescale(logvar[i], "LOGVAR")
        mu[i] = torch.tensor(mu[i], dtype=torch.float32).to(device)
        logvar[i] = torch.tensor(logvar[i], dtype=torch.float32).to(device)

    img_sequence = []
    for i in range(len(mu)):
        z = vae_model.reparameterize(mu[i], logvar[i])
        img = vae_model.decode(z).squeeze(0).cpu().detach().numpy()
        img_sequence.append(img)
    
    # Save the image sequence as a GIF
    image_list_to_gif(img_sequence, "output_sequence_true.gif", duration=100)


    print("Running inference with no teacher forcing...")
    # take a sample from the dataset
    sample_mu = sample_mu[:, :2, :].float().to(device)
    sample_logvar = sample_logvar[:, :2, :].float().to(device)
    sample_act = sample_act.float().to(device)
    mu, logvar = lstm_model.predict(sample_mu, sample_logvar, sample_act)

    for i in range(len(mu)):
        mu[i] = dataset.rescale(mu[i], "MU")
        logvar[i] = dataset.rescale(logvar[i], "LOGVAR")
        mu[i] = torch.tensor(mu[i], dtype=torch.float32).to(device)
        logvar[i] = torch.tensor(logvar[i], dtype=torch.float32).to(device)

    img_sequence = []
    for i in range(len(mu)):
        z = vae_model.reparameterize(mu[i], logvar[i])
        img = vae_model.decode(z).squeeze(0).cpu().detach().numpy()
        img_sequence.append(img)
    
    # Save the image sequence as a GIF
    image_list_to_gif(img_sequence, "output_sequence_sim.gif", duration=100)