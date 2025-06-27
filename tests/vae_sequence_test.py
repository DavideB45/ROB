import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

import random
from vae_model import VAE
from lstm_model import MuLogvarLSTM
from dataset.ordered_dataloader import Dataset
from helpers.utils_proj import device, image_list_to_gif
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

LATENT_DIM = 30

def get_model_VAE(latent_dim=LATENT_DIM):
	"""
	Load the VAE model with the specified latent dimension.
	"""
	model = VAE(latent_dim=latent_dim).to(device)
	model.load_state_dict(torch.load(f"models/vae_model_{latent_dim}_kl1_.pth", map_location=device))
	print(f"VAE Model # of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
	model.eval()  # Set the model to evaluation mode
	return model

def get_model_lstm(latent_dim=LATENT_DIM):
	"""
	Load the LSTM model with the specified latent dimension.
	"""
	lstm_model = MuLogvarLSTM(embedding_dim=latent_dim, hidden_dim=512, num_layers=2, dropout=0.1).to(device)
	lstm_model.load_state_dict(torch.load(f"models/lstm_vae_{latent_dim}.pth", map_location=device))
	print(f"LSTM Model # of parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)}")
	lstm_model.eval()  # Set the model to evaluation mode
	return lstm_model

def prepare_dataset(vae_model, seq_len=40):
	"""
	Create the dataset with the specified VAE model and sequence length.
	"""
	dataset = Dataset("dataset", "no_obj", vae_model, seq_len=seq_len)
	return dataset

def make_comp_gif(mu_pred:torch.Tensor, vae:VAE, ref:torch.Tensor, path:str, duration:int=800, dataset:Dataset=None):
	"""
	Create 2 GIF comparing the predicted sequence and the reference frames.
	"""
	mu_seq = []
	for i in range(len(mu_pred)):
		mu_seq.append(dataset.rescale(mu_pred[i], "MU"))
		mu_seq[i] = torch.from_numpy(mu_seq[i])
	
	mu_seq = torch.stack(mu_seq, dim=0)  # (seq_len, latent_dim)
	seq = vae.decode(mu_seq.to(device)).cpu()  # (seq_len, C, H, W)
	# Convert the sequence to a list of images
	image_list = [seq[i].cpu().detach() for i in range(seq.shape[0])]
	image_list_to_gif(image_list, path, duration=duration)
	image_list_ref = ref[0]
	image_list_ref = [image_list_ref[i].cpu().detach() for i in range(image_list_ref.shape[0])]
	image_list_to_gif(image_list_ref, path.replace(".gif", "_ref.gif"), duration=duration)

def compute_ssim(mu_pred:torch.Tensor, vae:VAE, ref:torch.Tensor):
	'''
	Compute the Mean Squared Error (MSE) between the predicted sequence and the reference frames.
	'''
	reconstructed_frame = vae.decode(mu_pred[-1].unsqueeze(0).to(device)).cpu()
	reconstructed_frame = torch.clamp(reconstructed_frame, 0, 1)
	reconstructed_frame = reconstructed_frame.permute(0, 2, 3, 1).squeeze()  # (1, H, W, C)
	last_frame_ref = ref[0][-1].cpu()#.permute(2, 0, 1)  # (C, H, W, C)
	#print(f"Reconstructed frame shape: {reconstructed_frame.shape}, Last frame reference shape: {last_frame_ref.shape}")
	ssim_value = ssim(reconstructed_frame.numpy(), last_frame_ref.numpy(), channel_axis=-1, data_range=1.0)
	return ssim_value

if __name__ == "__main__":

	# Create the dataset
	model_vae = get_model_VAE(latent_dim=LATENT_DIM)
	lstm_model = get_model_lstm(latent_dim=LATENT_DIM)

	mse_results = []

	for seq_len in range(3, 21):
		# Get training and validation sets
		dataset = prepare_dataset(model_vae, seq_len=seq_len)
		ts = torch.utils.data.DataLoader(dataset.get_test_set(), batch_size=1, shuffle=False)
		ts_ref = torch.utils.data.DataLoader(dataset.get_test_set_ref(), batch_size=1, shuffle=False)
		average_mse = 0.0
		with torch.no_grad():
			for (mu_frames, logvar_frames, act_frames), ref in zip(ts, ts_ref):
				mu, log = lstm_model.predict(
					mu_frames.to(device),
					logvar_frames.to(device),
					act_frames.float().to(device)
				)
				mse_error = compute_ssim(mu, model_vae, ref)
				average_mse += mse_error
				#make_comp_gif(mu, model_vae, ref, f"figures/comp_0.gif", dataset=dataset)
				#exit(0)
		average_mse /= len(ts)
		mse_results.append(average_mse)
		print(f"Seq_len: {seq_len}, Average SSIM: {average_mse:.4f}")

	print("SSIM results for sequence lengths 3 to 20:", mse_results)

	# Plotting
	plt.figure(figsize=(8, 5))
	plt.plot(range(3, 21), mse_results, marker='o')
	plt.xlabel('Sequence Length')
	plt.ylabel('Average SSIM')
	plt.title('SSIM vs Sequence Length')
	plt.grid(True)
	plt.show()
