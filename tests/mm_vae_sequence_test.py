import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

import random
from moe_vae import load_mmvae_model, VAE
from lstm_model import MuLogvarLSTM
from dataset.m_dataloader import MDataset
from helpers.utils_proj import device, image_list_to_gif
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

LATENT_DIM = 200
POS_W = False

def get_model_MMVAE(latent_dim=LATENT_DIM):
	"""
	Load the VAE model with the specified latent dimension.
	"""
	model = load_mmvae_model(f"models/moe_vae_model_{latent_dim}{"_" if POS_W else ""}.pth", 20, latent_dim).to(device)
	print(f"MMVAE Model # of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
	model.eval()  # Set the model to evaluation mode
	return model

def get_model_lstm(latent_dim=LATENT_DIM):
	"""
	Load the LSTM model with the specified latent dimension.
	"""
	lstm_model = MuLogvarLSTM(embedding_dim=latent_dim, hidden_dim=512, num_layers=2, dropout=0.1).to(device)
	lstm_model.load_state_dict(torch.load(f"models/lstm_mmvae_{latent_dim}{"_" if POS_W else ""}.pth", map_location=device))
	print(f"LSTM Model # of parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)}")
	lstm_model.eval()  # Set the model to evaluation mode
	return lstm_model

def get_dataset():
	"""
	Create the dataset with the specified VAE model and sequence length.
	"""
	dataset = MDataset("dataset", "no_obj")
	return dataset

def make_comp_gif(mu_pred:torch.Tensor, vae:VAE, ref:torch.Tensor, path:str, duration:int=800, dataset:MDataset=None):
	"""
	Create 2 GIF comparing the predicted sequence and the reference frames.
	"""
	mu_seq = []
	for i in range(len(mu_pred)):
		mu_seq.append(dataset.rescale(mu_pred[i].cpu().detach(), "MU"))
		mu_seq[i] = torch.from_numpy(mu_seq[i])
	
	mu_seq = torch.stack(mu_seq, dim=0)  # (seq_len, latent_dim)
	seq = vae.decode(mu_seq.to(device)).cpu()  # (seq_len, C, H, W)
	# Convert the sequence to a list of images
	image_list = [seq[i].cpu().detach() for i in range(seq.shape[0])]
	image_list_to_gif(image_list, path, duration=duration)
	image_list_ref = ref[0]
	image_list_ref = [image_list_ref[i].cpu().detach() for i in range(image_list_ref.shape[0])]
	image_list_to_gif(image_list_ref, path.replace(".gif", "_ref.gif"), duration=duration)

def compute_ssim(mu_pred:torch.Tensor, vae:VAE, ref:np.ndarray):
	'''
	Compute the Mean Squared Error (MSE) between the predicted sequence and the reference frames.
	'''
	reconstructed_frame = vae.decode(mu_pred[-1].unsqueeze(0).to(device)).cpu()
	reconstructed_frame = torch.clamp(reconstructed_frame, 0, 1)
	reconstructed_frame = reconstructed_frame.permute(0, 2, 3, 1).squeeze()  # (1, H, W, C)
	last_frame_ref = ref[0][-1].cpu().permute(1,2,0)  # (C, H, W, C)
	#print(f"Reconstructed frame shape: {reconstructed_frame.shape}, Last frame reference shape: {last_frame_ref.shape}")
	ssim_value = ssim(reconstructed_frame.numpy(), last_frame_ref.numpy(), channel_axis=-1, data_range=1.0)
	return ssim_value

if __name__ == "__main__":
	print("\033[93mMMVAE Sequence Length Test, Latent Dim:", LATENT_DIM, "POS_W:", POS_W, "\033[0m")
	# Create the dataset
	model_mmvae = get_model_MMVAE(latent_dim=LATENT_DIM)
	lstm_model = get_model_lstm(latent_dim=LATENT_DIM)
	dataset = get_dataset()

	mse_results = []

	#num = 0
	for seq_len in range(30, 31):
		dataset.prepare_hidden_sequence(model_mmvae, seq_len=seq_len, device=device)
		dataset.get_sequence_loaders(test_perc=0.3)
		# Get training and validation sets
		ts = dataset.get_test_set()
		ts_ref = dataset.get_visual_reference()
		#print(f"Testing with sequence length: {seq_len}, Number of sequences: {len(ts)}")
		#print(f"Number of reference frames: {len(ts_ref)}")
		average_mse = 0.0
		with torch.no_grad():
			for (mu_frames, logvar_frames, act_frames), ref in zip(ts, ts_ref):
				# if num < 37:
				# 	num += 1
				# 	continue
				mu, log = lstm_model.predict(
					mu_frames[:,:2,:].to(device),
					logvar_frames[:,:2,:].to(device),
					act_frames.float().to(device)
				)
				mse_error = compute_ssim(mu, model_mmvae.vision, ref)
				average_mse += mse_error
				make_comp_gif(mu, model_mmvae.vision, ref, f"figures/comp_mmvae_0_{LATENT_DIM}.gif", dataset=dataset)
				exit(0)
		average_mse /= len(ts)
		mse_results.append(average_mse)
		print(f"Seq_len: {seq_len}, Average SSIM: {average_mse:.4f}")

	print("\033[92mSSIM results for sequence lengths 3 to 20:", mse_results, "\033[0m")

	# Plotting
	plt.figure(figsize=(8, 5))
	plt.plot(range(3, 31), mse_results, marker='o')
	plt.xlabel('Sequence Length')
	plt.ylabel('Average SSIM')
	plt.title('SSIM vs Sequence Length')
	plt.grid(True)
	plt.show()
