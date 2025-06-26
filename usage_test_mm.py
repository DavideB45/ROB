from moe_vae import MoE_VAE, load_mmvae_model
from dataset.m_dataloader import MDataset
import torch
from lstm_model import MuLogvarLSTM
from helpers.utils_proj import device, image_list_to_gif
import random

BATCH_SIZE = 128
LATENT_DIM = 200

dataset = MDataset("./dataset", "no_obj")
model = load_mmvae_model(f"./models/moe_vae_model_{LATENT_DIM}.pth", 20, LATENT_DIM).to(device)
model.eval()
dataset.prepare_hidden_sequence(model, seq_len=40, device=device)
_, vl = dataset.get_sequence_loaders(test_perc=0.3, batch_size=1, shuffle=False)
vl_ref = dataset.get_visual_reference()

lstm_model = MuLogvarLSTM(embedding_dim=LATENT_DIM, hidden_dim=512, num_layers=2, dropout=0.1)
lstm_model.load_state_dict(torch.load(f"./models/lstm_model_mm_{LATENT_DIM}.pth", map_location=device))
lstm_model.eval()

i = random.randint(0, len(vl) - 1)  # Randomly select a sample index
#i = 5
sample_mu, sample_logvar, sample_act = list(vl)[i]
sample_mu.to(device), sample_logvar.to(device), sample_act.to(device)
print(f"Using sample index: {i} of {len(vl)}")
sample_img = list(vl_ref)[i]

mu, logvar = lstm_model.predict(sample_mu, sample_logvar, sample_act)
for i in range(len(mu)):
    mu[i] = dataset.rescale(mu[i], "MU")
    logvar[i] = dataset.rescale(logvar[i], "LOGVAR")
    mu[i] = torch.tensor(mu[i], dtype=torch.float32).to(device)
    logvar[i] = torch.tensor(logvar[i], dtype=torch.float32).to(device)
img_sequence_sensed = []
for i in range(len(mu)):
    z = model.reparameterize(mu[i], logvar[i])
    img, _ = model.decode(z)
    img = img.squeeze(0).cpu().detach().numpy()
    img_sequence_sensed.append(img)

sample_mu = sample_mu[:, :4, :].float().to(device)
sample_logvar = sample_logvar[:, :4, :].float().to(device)
sample_act = sample_act.to(device)
lstm_model.to(device)
mu, logvar = lstm_model.predict(sample_mu, sample_logvar, sample_act)
for i in range(len(mu)):
    mu[i] = dataset.rescale(mu[i].cpu(), "MU")
    logvar[i] = dataset.rescale(logvar[i].cpu(), "LOGVAR")
    mu[i] = torch.tensor(mu[i], dtype=torch.float32).to(device)
    logvar[i] = torch.tensor(logvar[i], dtype=torch.float32).to(device)
img_sequence_sim = []
for i in range(len(mu)):
    z = model.reparameterize(mu[i], logvar[i])
    img, _ = model.decode(z)
    img = img.squeeze(0).cpu().detach().numpy()
    img_sequence_sim.append(img)

img_seqence_ref = []
# shape is (batch_size, seq_len, c, h, w)
print(f"Sample image shape: {sample_img[0].shape}")
for i in range(sample_img[0].shape[0]):
    img = sample_img[0][i].cpu().detach().numpy()
    img_seqence_ref.append(img)


# Save the image sequence as a GIF
image_list_to_gif(img_sequence_sensed[:], "output_sequence_sensed_mm.gif", duration=200)
# Save the image sequence as a GIF
image_list_to_gif(img_sequence_sim[:], "output_sequence_sim_mm.gif", duration=200)
# Save the image sequence as a GIF
image_list_to_gif(img_seqence_ref[:], "output_sequence_ref_mm.gif", duration=200)