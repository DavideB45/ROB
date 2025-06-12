import torch
from vae import VAE
from dataset import Dataset
from utils_proj import device
dataset:Dataset = Dataset(
	path="dataset",
	condition="no_obj"
)

def main(name: str = "vae_model_final.pth"):

	model = VAE(latent_dim=200).to(device)
	model.load_state_dict(torch.load(name))