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
from helpers.utils_proj import get_best_device

device = get_best_device()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir:str, ds_cond:str, vision_vae:VAE, seq_len:int=10):
        # self.data will be a list of tuples (mu_frames, logvar_frames, act_frames)
        # where each tuple contains 10 frames of vision data and corresponding actions
        # vision start at time t, act at time t-1 to have idea of dynamics
        self.data = None
        self.seq_len = seq_len  # Length of the sequence to consider for each sample
        # Get simulation data
        pos,force,vision,act = self.get_data(ds_dir, ds_cond)
        # they have len 7322
        print(f"vision shape: {vision.shape}, act shape: {act.shape}")
        # remove 2 frames (to be divisible by 10)
        pos = pos[1:-1]
        vision = vision[1:-1]
        act = act[:-2] # different to account for previous action
        print(f"pos shape: {pos.shape}, vision shape: {vision.shape}, act shape: {act.shape}")
        self.setup_dataset(vision, act, vision_vae)
        self.split_data(vision)
        
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
        """Setup the dataset by generating training dataset. And rescaling the data.
        This method encodes the vision data using a VAE, divides the dataset into sections of 10 frames,
        Args:
            vision (np.ndarray): The vision data.
            act (np.ndarray): The action data.
            vision_vae (VAE): The VAE model for vision data.
        """
        # encode the dataset to get input of the LSTM
        print("Setting up dataset...")
        vision = torch.tensor(vision, dtype=torch.float32)  # Convert to tensor
        vision = vision.permute(0, 3, 1, 2).contiguous().to(device)  # Change to (batch_size, channels, height, width)
        with torch.no_grad():
            world_embedding = vision_vae.encode(vision)
        #memory efficient encoding
        # divide the dataset into section of duration 10 frames
        self.data = []
        self.sc_MU = StandardScaler()
        world_embedding_mu = self.sc_MU.fit_transform(world_embedding[0].detach().cpu().numpy())
        self.sc_LOGVAR = StandardScaler()
        world_embedding_lv = self.sc_LOGVAR.fit_transform(world_embedding[1].detach().cpu().numpy())
        for i in range(0, len(act) - self.seq_len, self.seq_len):
            mu_frames = world_embedding_mu[i:i + self.seq_len]  # take 10 frames of mean
            logvar_frames = world_embedding_lv[i:i + self.seq_len]
            act_frames = act[i:i + self.seq_len]  # take 10 frames of actions
            if len(mu_frames) == self.seq_len and len(act_frames) == self.seq_len:
                self.data.append((mu_frames, logvar_frames, act_frames))
            else:
                print(f"Skipping incomplete frame set at index {i}: {len(mu_frames)} frames, {len(act_frames)} actions")  
    
    def split_data(self, vision:np.ndarray) -> None:
        """Split the dataset into training and validation sets."""
        #self.tr, self.vs = train_test_split(self.data, test_size=0.1, random_state=42)
        split_idx = int(len(self.data) * 0.9)
        self.tr = self.data[:split_idx]
        self.vs = self.data[split_idx:]
        vision_blocked = []
        for i in range(0, len(vision) - self.seq_len, self.seq_len):
            vision_blocked.append(vision[i:i + self.seq_len])
        vision_blocked = np.array(vision_blocked)
        self.tr_ref = vision_blocked[:split_idx]
        self.vs_ref = vision_blocked[split_idx:]
 
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
        elif type == "MU":
            sc = self.sc_MU
        elif type == "LOGVAR":
            sc = self.sc_LOGVAR
        return sc.inverse_transform(data.detach().cpu().numpy())
    
    def get_training_set(self):
        return self.tr
    
    def get_training_set_ref(self):
        return self.tr_ref
    
    def get_validation_set(self):
        return self.vs
    
    def get_validation_set_ref(self):
        return self.vs_ref
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.tr[index][0], self.tr[index][1], self.tr[index][2]  # Return mu, logvar, act frames

if __name__ == "__main__":
    # Example usage
    ds_dir = "dataset"
    ds_cond = "no_obj"
    vision_vae = VAE(latent_dim=200).to(device)  # Assuming you have a pre-trained VAE model
    vision_vae.load_state_dict(torch.load("models/vae_final_model.pth", map_location=device))
    vision_vae.eval()  # Set the VAE to evaluation mode
    dataset = Dataset(ds_dir, ds_cond, vision_vae)
    
    print(f"Dataset length: {len(dataset)}")

    # Example of accessing a sample
    sample_mu, sample_logvar, sample_act = dataset[0]
    print(f"Sample mu shape: {sample_mu.shape}, logvar shape: {sample_logvar.shape}, act shape: {sample_act.shape}")
    print("FAKE TRAINING")
    train_loader = torch.utils.data.DataLoader(dataset.get_training_set(), batch_size=16, shuffle=True)
    for i, (mu_frames, logvar_frames, act_frames) in enumerate(train_loader):
        print(f"Batch {i+1}: mu_frames shape: {mu_frames.shape}, logvar_frames shape: {logvar_frames.shape}, act_frames shape: {act_frames.shape}")
        if i == 5:  # Just to limit output for demonstration
            print(f"logvars:")
            for j in range(len(logvar_frames[0])):
                print(f"  Sample 0 frame {j+1}: {logvar_frames[0][j][5:10].cpu().detach().numpy()}")
            break