import random
from vae_model import VAE
from lstm_model import MuLogvarLSTM
from dataset.ordered_dataloader import Dataset
from helpers.utils_proj import get_best_device, image_list_to_gif
import torch

if __name__ == "__main__":

    # Set the device to CUDA if available, otherwise CPU
    device = get_best_device()

    # Initialize the VAE model
    latent_dim = 200
    vae_model = VAE(latent_dim=latent_dim).to(device)
    "models/vae_final_model.pth"
    "vae_model_foundation_kl04_l2e4_ed200.pth"
    "vae_model_foundation_kl04_l3e4_ed90.pth"
    vae_model.load_state_dict(torch.load("models/vae_model_foundation_kl04_l2e4_ed200.pth", map_location=device))
    vae_model.eval()  # Set the model to evaluation mode

    # Create the dataset
    dataset = Dataset("dataset", "no_obj", vae_model, seq_len=44)

    # Get training and validation sets
    train_set = dataset.get_validation_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    train_set_ref = dataset.get_validation_set_ref()
    train_set_ref_loader = torch.utils.data.DataLoader(train_set_ref, batch_size=1, shuffle=False)
    i = len(train_set) // 2  # Get a sample from the middle of the dataset
    i = random.randint(0, len(train_set) - 1)  # Randomly select a sample index
    print(f"Using sample index: {i} of {len(train_set)}")
    sample_mu, sample_logvar, sample_act = list(train_loader)[i]
    sample_img = list(train_set_ref_loader)[i]

    # Initialize the LSTM model
    lstm_model = MuLogvarLSTM(embedding_dim=latent_dim, hidden_dim=512, num_layers=2, dropout=0.1).to(device)
    lstm_model.load_state_dict(torch.load("models/lstm_final_model_200.pth", map_location=device))
    print(f"Model # of parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)}")
    lstm_model.eval()  # Set the model to evaluation mode

    print("Running inference with teacher forcing...")
    # take a sample from the dataset
    sample_mu = sample_mu.float().to(device)
    sample_logvar = sample_logvar.float().to(device)
    sample_act = sample_act.float().to(device)
    mu, logvar = lstm_model.predict(sample_mu, sample_logvar, sample_act)

    for i in range(len(mu)):
        mu[i] = dataset.rescale(mu[i], "MU")
        logvar[i] = dataset.rescale(logvar[i], "LOGVAR")
        mu[i] = torch.tensor(mu[i], dtype=torch.float32).to(device)
        logvar[i] = torch.tensor(logvar[i], dtype=torch.float32).to(device)

    img_sequence_sensed = []
    for i in range(len(mu)):
        z = vae_model.reparameterize(mu[i], logvar[i])
        img = vae_model.decode(z).squeeze(0).cpu().detach().numpy()
        img_sequence_sensed.append(img)
    
    


    print("Running inference with no teacher forcing...")
    # take a sample from the dataset
    sample_mu = sample_mu[:, :4, :].float().to(device)
    sample_logvar = sample_logvar[:, :4, :].float().to(device)
    sample_act = sample_act.float().to(device)
    mu, logvar = lstm_model.predict(sample_mu, sample_logvar, sample_act)

    for i in range(len(mu)):
        mu[i] = dataset.rescale(mu[i], "MU")
        logvar[i] = dataset.rescale(logvar[i], "LOGVAR")
        mu[i] = torch.tensor(mu[i], dtype=torch.float32).to(device)
        logvar[i] = torch.tensor(logvar[i], dtype=torch.float32).to(device)

    img_sequence_sim = []
    for i in range(len(mu)):
        z = vae_model.reparameterize(mu[i], logvar[i])
        img = vae_model.decode(z).squeeze(0).cpu().detach().numpy()
        img_sequence_sim.append(img)
    
    img_sequence_ref = []
    # shape is (batch_size, seq_len, w, h, c)
    print(f"Sample image shape: {sample_img[0].shape}")
    for i in range(sample_img[0].shape[0]):
        img = sample_img[0][i]
        img = img.permute(2, 0, 1)
        img_sequence_ref.append(img)

    # Save the image sequence as a GIF
    image_list_to_gif(img_sequence_ref[:], "output_sequence_ref.gif", duration=100)
    # Save the image sequence as a GIF
    image_list_to_gif(img_sequence_sensed[:], "output_sequence_sensed.gif", duration=100)
    # Save the image sequence as a GIF
    image_list_to_gif(img_sequence_sim[:], "output_sequence_sim.gif", duration=100)
