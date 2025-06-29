import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

import random
from moe_vae import load_mmvae_model, VAE, FC_VAE
from lstm_model import MuLogvarLSTM
from dataset.m_dataloader import MDataset
from helpers.utils_proj import device, image_list_to_gif
from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np
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

def make_comp_gif(mu_pred:torch.Tensor, vae:VAE, ref:torch.Tensor, path:str, duration:int=200, dataset:MDataset=None):
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
	reconstructed_frame = vae.decode(mu_pred.to(device)).cpu()
	reconstructed_frame = torch.clamp(reconstructed_frame, 0, 1)
	reconstructed_frame = reconstructed_frame.permute(0, 2, 3, 1).squeeze()  # (1, H, W, C)
	last_frame_ref = ref.cpu().permute(1,2,0)  # (C, H, W, C)
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
	for seq_len in range(3, 31):
		raw_sequences = dataset.get_raw_sequence(seq_len)
		average_mse = 0.0
		with torch.no_grad():
			for (vis, pos, act) in raw_sequences:
				# if num < 30:
				# 	num += 1
				# 	continue
				# Prepare the input tensors
				vis = vis.to(device)
				pos = pos.to(device)
				act = act.to(device)
				#print(f"Processing sequence length {seq_len} with shape: vis={vis.shape}, pos={pos.shape}, act={act.shape}")
				# Get the latent representations fro the first two frames
				#print(f"Shape pa")
				mu_early, logvar_early = model_mmvae.encode(vis[:2,:,:,:], pos[:2,:])
				#print(f"Shape mu_early: {mu_early.shape}, logvar_early: {logvar_early.shape}")
				outputs_mu = []
				outputs_logvar = []
				lstm_model.eval()  # Ensure the LSTM model is in evaluation mode
				for t in range(seq_len):
					if t < 2:
						mu_input = mu_early[t, :]
						logvar_input = logvar_early[t, :]
					else:
						pos_decoded = model_mmvae.proprioception.decode(outputs_mu[-1])
						mu_input = mu_input - model_mmvae.proprioception.encode(pos_decoded)[0].squeeze(0) + model_mmvae.proprioception.encode(pos[t, :].unsqueeze(0))[0].squeeze(0)
						logvar_input = torch.zeros_like(mu_input)
					act_t = act[t, :]
					act_t1 = act[t + 1, :]
					lstm_input = torch.cat([mu_input, logvar_input, act_t, act_t1], dim=-1)
					#print(f"Shape lstm_input: {lstm_input.shape}")
					mu_pred, logvar_pred, _ = lstm_model.single_element_forward(lstm_input)
					outputs_mu.append(mu_pred)  # Transpose to match the expected shape
					outputs_logvar.append(logvar_pred)
				# Compute 
				mse_error = compute_ssim(outputs_mu[-1], model_mmvae.vision, vis[-1])
				average_mse += mse_error
				#make_comp_gif(outputs_mu, model_mmvae.vision, ref, f"figures/comp_mmvae_0_{LATENT_DIM}.gif", dataset=dataset)
				#exit(0)
		average_mse /= len(raw_sequences)
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
