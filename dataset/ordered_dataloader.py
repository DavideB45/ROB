import torch
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import pickle
import matplotlib.pyplot as plt
import random

sys.path.append(os.path.dirname(os.path.dirname("..")))

from vae import VAE
from utils_proj import get_best_device

device = get_best_device()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir:str, ds_cond:str, vision_vae:VAE):
        # self.data will be a list of tuples (mu_frames, logvar_frames, act_frames)
        # where each tuple contains 10 frames of vision data and corresponding actions
        # vision start at time t, act at time t-1 to have idea of dynamics
        self.data = None
        # Get simulation data
        pos,force,vision,act = self.get_data(ds_dir, ds_cond)
        # they have len 7322
        print(f"vision shape: {vision.shape}, act shape: {act.shape}")
        # remove 2 frames (to be divisible by 10)
        pos = pos[1:-1]
        vision = vision[1:-1]
        act = act[:-2] # different to account for previous action
        self.setup_dataset(vision, act, vision_vae)
        self.split_data()
        
    def get_data(self, path:str, condition:str) -> tuple:
        """Load and preprocess data from the specified path and condition.
        Args:
            path (str): The path to the directory containing the data files.
            condition (str): The condition of the trial (e.g., "obj", "no_obj").
        Returns:
            tuple: A tuple containing the preprocessed data (POS, FORCE, VISION, ACT).
        """                
        # Load data
        _, pos, force, act, vision = self.load_data(path,0,condition)
        # Data pre-processing
        data = (pos,force,vision,act)
        return self.preprocess_data(data)
        
    def load_data(self,path:str,trial:int,condition:str) -> tuple:
        """Load data from a gzip file ending with .pkl.

        Args:
            path (str): The path to the directory containing the data files.
            trial (int): The trial number (0-indexed).
            condition (str): The condition of the trial (e.g., "obj", "no_obj").
        
        Returns:
            tuple: A tuple containing the loaded data (time, pos, force, act, vision).
        """
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
        return (time,pos,force,act,vision)
    
    def preprocess_data(self,data:tuple) -> tuple:
        """Preprocess the data by scaling and normalizing.
        Args:
            data (tuple): A tuple containing the raw data (pos, force, vision, act).
        Returns:
            tuple: A tuple containing the preprocessed data (pos, force, vision, act).
        """
        pos,force,vision,act = data
        # Process data with scaling (standardized scaling)
        self.sc_POS = StandardScaler()
        pos = self.sc_POS.fit_transform(pos)
        self.sc_FORCE = StandardScaler()
        force = self.sc_FORCE.fit_transform(force)
        self.sc_ACT = StandardScaler()
        act = self.sc_ACT.fit_transform(act)
        vision = vision/255.        
        return (pos,force,vision,act)
    
    def setup_dataset(self, vision, act, vision_vae:VAE) -> None:
        """Setup the dataset by generating training dataset.
        Args:
            vision (np.ndarray): The vision data.
            act (np.ndarray): The action data.
            vision_vae (VAE): The VAE model for vision data.
        """
        # encode the dataset to get input of the LSTM
        vision = torch.tensor(vision, dtype=torch.float32).to(device)
        vision = vision.permute(0, 3, 1, 2).contiguous()  # Change to (batch_size, channels, height, width)
        world_embedding = vision_vae.encode(vision)
        # divide the dataset into section of duration 10 frames
        self.data = []
        print(f"world_embedding shape: {world_embedding[0].shape}, act shape: {act.shape}")
        for i in range(0, len(act) - 10, 10):
            mu_frames = world_embedding[0][i:i + 10]  # take 10 frames of mean
            logvar_frames = world_embedding[1][i:i + 10]
            act_frames = act[i:i + 10]  # take 10 frames of actions
            if len(mu_frames) == 10 and len(act_frames) == 10:
                self.data.append((mu_frames, logvar_frames, act_frames))
            else:
                print(f"Skipping incomplete frame set at index {i}: {len(mu_frames)} frames, {len(act_frames)} actions")       
    
    def split_data(self):
        """Split the dataset into training and validation sets."""
        # Convert data to tensors
        self.tr, self.vs = train_test_split(self.data, test_size=0.3, random_state=42)
 
    def rescale(self, data:torch.Tensor, type:str) -> np.ndarray:
        """Rescale the data based on its type.
        Args:
            data (torch.Tensor): The data to be rescaled.
            type (str): The type of data ("VISION", "FLOW", "POS", "FORCE", "ACT").
        Returns:
            numpy.ndarray: The rescaled data.
        """
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
    
    def get_training_set(self):
        return self.ts
    
    def get_validation_set(self):
        return self.vs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.tr[index][0], self.tr[index][1], self.tr[index][2]  # Return mu, logvar, act frames

if __name__ == "__main__":
    # Example usage
    ds_dir = "dataset"
    ds_cond = "no_obj"
    vision_vae = VAE(latent_dim=200).to(device)  # Assuming you have a pre-trained VAE model
    vision_vae.load_state_dict(torch.load("vae_final_model.pth", map_location=device))
    vision_vae.eval()  # Set the VAE to evaluation mode
    dataset = Dataset(ds_dir, ds_cond, vision_vae)
    
    print(f"Dataset length: {len(dataset)}")

    # Example of accessing a sample
    sample_mu, sample_logvar, sample_act = dataset[0]
    print(f"Sample mu shape: {sample_mu.shape}, logvar shape: {sample_logvar.shape}, act shape: {sample_act.shape}")