import torch
import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import pickle
sys.path.append(os.path.dirname(os.path.dirname("..")))
from moe_vae import MoE_VAE, create_mmvae_model

class MDataset(torch.utils.data.Dataset):
	def __init__(self, path, condition):
		# Get simulation data
		position,FORCE,vision,cond_input = self.get_data(path, 0, condition)
		
		self.position = position
		self.vision = vision
		self.cond_input = cond_input
		self.FORCE = FORCE
				
	def get_data(self, path, trials, condition):                
		for trial in range(trials+1):
			# Load data
			_,pos,force,act,vision = self.load_data(path,trial +1,condition)
			# Merge trials (if more than 1)
			if trial == 0:
				POS = pos
				FORCE = force
				VISION = vision
				ACT = act
			else:
				POS = np.vstack((POS,pos))
				FORCE = np.vstack((FORCE,force))
				VISION = np.vstack((VISION,vision))
				ACT = np.vstack((ACT,act))
		# Data pre-processing
		data = (POS,FORCE,VISION,ACT)
		return self.preprocess_data(data)
		
	def load_data(self,path,trial,condition):
		# Build the complete path and open the file
		path_final = os.path.join(path,"trial_"+str(trial+1)+"_"+condition+".pkl")
		print("Loading data from: ", path_final)
		f = gzip.open(path_final,"rb")
		# Read the file contents
		time = pickle.load(f)
		pos = pickle.load(f)
		force = pickle.load(f)
		act = pickle.load(f)
		vision = pickle.load(f)
		# Close the file
		f.close()
		print(f"Data loaded: time={len(time)}, pos={pos.shape}, force={force.shape}, act={act.shape}, vision={vision.shape}")
		return (time,pos,force,act,vision)
	
	def preprocess_data(self,data):
		POS,FORCE,VISION,ACT = data
		# Process data with scaling (standardized scaling)
		self.sc_POS = StandardScaler()
		POS = self.sc_POS.fit_transform(POS)
		self.sc_FORCE = StandardScaler()
		FORCE = self.sc_FORCE.fit_transform(FORCE)
		self.sc_ACT = StandardScaler()
		ACT = self.sc_ACT.fit_transform(ACT)
		VISION = VISION/255.
		# Convert to torch tensors
		POS = torch.tensor(POS, dtype=torch.float32)
		FORCE = torch.tensor(FORCE, dtype=torch.float32)
		VISION = torch.tensor(VISION, dtype=torch.float32)
		ACT = torch.tensor(ACT, dtype=torch.float32)
		# Reshape VISION to match the expected input shape (N, C, H
		VISION = VISION.permute(0, 3, 1, 2).contiguous()
		return (POS,FORCE,VISION,ACT)
	
	def get_VAE_loaders(self, types:list[str], test_perc:float = 0.1, batch_size:int=32, shuffle:bool=True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
		"""Get DataLoaders for VAE training."""
		sets = {}
		for type in types:
			if type == "VISION":
				data = self.vision
			elif type == "POS":
				data = self.position
			elif type == "FORCE":
				data = self.FORCE
			elif type == "ACT":
				data = self.cond_input
			else:
				raise ValueError(f"Unknown data type: {type}")
			sets[type] = data

		final_dataset = []
		for i in range(len(sets[types[0]])):
			final_dataset.append(tuple(sets[type][i] for type in types))
		X_train, X_val = train_test_split(final_dataset, test_size=test_perc, random_state=42)
		train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=shuffle)
		val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False)
		return train_loader, val_loader

	def prepare_hidden_sequence(self, mmvae: MoE_VAE, seq_len:int, device: str = "cpu"):
		"""Prepare the hidden sequence for the LSTM."""
		#encode values
		with torch.no_grad():
			mmvae.eval()
			mu_list, logvar_list = [], []
			batch_size = 16
			num_samples = self.vision.shape[0]
			for i in range(0, num_samples, batch_size):
				vision_batch = self.vision[i:i+batch_size].to(device)
				pos_batch = self.position[i:i+batch_size].to(device)
				mu_batch, logvar_batch = mmvae.encode(vision_batch, pos_batch)
				mu_list.append(mu_batch.cpu())
				logvar_list.append(logvar_batch.cpu())
				del vision_batch, pos_batch, mu_batch, logvar_batch
				torch.mps.empty_cache()
			mu = torch.cat(mu_list, dim=0)
			logvar = torch.cat(logvar_list, dim=0)
		# Rescale mu and logvar
		self.sc_MU = StandardScaler()
		mu = self.sc_MU.fit_transform(mu.detach().cpu().numpy())
		self.sc_LOGVAR = StandardScaler()
		logvar = self.sc_LOGVAR.fit_transform(logvar.detach().cpu().numpy())
		# remove the first frame since we need previous action for LSTM input
		mu = mu[1:]
		logvar = logvar[1:]
		self.sequence = []
		self.visual_ref = []
		for i in range(0, len(mu) - seq_len, seq_len):
			mu_frames = mu[i:i + seq_len]
			logvar_frames = logvar[i:i + seq_len]
			act_frames = self.cond_input[i:i + seq_len]
			if len(mu_frames) == seq_len and len(act_frames) == seq_len:
				self.sequence.append((mu_frames, logvar_frames, act_frames))
				self.visual_ref.append(self.vision[i+1:i+1 + seq_len])
			else:
				print(f"Skipping incomplete frame set at index {i}: {len(mu_frames)} frames, {len(act_frames)} actions")
		self.seq_len = seq_len

	def get_sequence_loaders(self, test_perc: float = 0.1, batch_size: int = 32, shuffle: bool = True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
		"""Get DataLoaders for the prepared hidden sequence."""
		if not hasattr(self, 'sequence'):
			raise ValueError("Hidden sequence not prepared. Call prepare_hidden_sequence first.")
		
		split_idx = int(len(self.sequence) * (1 - test_perc))
		split_idx2 = int(len(self.visual_ref) * (1 - test_perc/3))
		train_data = self.sequence[:split_idx]
		val_data = self.sequence[split_idx:split_idx2]
		test_data = self.sequence[split_idx2:]
		visual_ref_val = self.visual_ref[split_idx:split_idx2]
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
		val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

		self.visual_ref = []
		for i in range(0, len(visual_ref_val), batch_size):
			self.visual_ref.append(visual_ref_val[i:i + batch_size])
		return train_loader, val_loader
	
	def get_visual_reference(self) -> np.ndarray:
		"""Get visual reference for validation."""
		if not hasattr(self, 'visual_ref'):
			raise ValueError("Visual reference not prepared. Call get_sequence_loaders first.")
		return self.visual_ref
		

	def rescale(self, data, type):
		if type == "VISION" or type == "FLOW":
			data = data*255.
			return data
		elif type == "POS":
			sc = self.sc_POS
		elif type == "FORCE":
			sc = self.sc_FORCE
		elif type == "ACT":
			sc = self.sc_ACT
		elif type == "MU":
			sc = self.sc_MU
		elif type == "LOGVAR":
			sc = self.sc_LOGVAR
		return sc.inverse_transform(data.detach().numpy())
	
	def __len__(self):
		return len(self.self.vision)
	
	def __getitem__(self, index):
		return (self.vision[index],
				self.position[index],
				self.FORCE[index],
				self.cond_input[index])
	
def main():
	# Path to data location
	path = "./dataset"
	scenario = "no_obj"
	
	# Build the training and validation datasets
	feature_set = MDataset(path,scenario)

	tr, vl = feature_set.get_VAE_loaders(["VISION", "POS", "FORCE", "ACT"], test_perc=0.1, batch_size=32, shuffle=True)
	print("Training DataLoader created with", len(tr.dataset), "batches.")
	for vis, pos, forc, act in tr:
		print("Vision batch shape:", vis.shape)
		print("Position batch shape:", pos.shape)
		print("Force batch shape:", forc.shape)
		print("Action batch shape:", act.shape)
		break  # Just to print the first batch

	print("Validation DataLoader created with", len(vl.dataset), "batches.")
	for vis, pos, forc, act in vl:
		print("Validation Vision batch shape:", vis.shape)
		print("Validation Position batch shape:", pos.shape)
		print("Validation Force batch shape:", forc.shape)
		print("Validation Action batch shape:", act.shape)
		break

	print()

	# Prepare the hidden sequence for the LSTM
	latent_dim = 20
	mmvae = create_mmvae_model(
		proprioception_input_dim=20,
		latent_dim=latent_dim
	)
	feature_set.prepare_hidden_sequence(mmvae, seq_len=10, device="cpu")
	# Get the sequence loaders
	train_seq_loader, val_seq_loader = feature_set.get_sequence_loaders(test_perc=0.1, batch_size=32, shuffle=True)
	print("Training sequence len :", len(train_seq_loader.dataset), "sequences.", "Validation sequence len :", len(val_seq_loader.dataset), "sequences.")
	for mu, logvar, act in train_seq_loader:
		print("Mu frames batch shape:", mu.shape)
		print("Logvar frames batch shape:", logvar.shape)
		print("Action frames batch shape:", act.shape)
		break

	val_ref = feature_set.get_visual_reference()
	print("Validation visual reference created with", len(val_ref), "batches.")
	for vis in val_ref:
		print("Validation visual reference batch len:", len(vis), "sequences.")
		print("First sequence visual reference shape:", vis[0].shape)
		break
	
if __name__ == "__main__":
    main()