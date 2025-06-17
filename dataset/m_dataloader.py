import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import pickle
import matplotlib.pyplot as plt
import random
import cv2

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
			_,pos,force,act,vision = self.load_data(path,trial,condition)
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
	
if __name__ == "__main__":
    main()