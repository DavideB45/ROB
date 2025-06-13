from torch.nn import LSTM, Linear, BatchNorm1d, ReLU
from torch.nn.functional import relu, mse_loss
import torch.nn as nn
import torch

class MuLogvarLSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 1, 
                 dropout: float = 0.2):
        """
        Initializes the MuLogvarLSTM model.
        Args:
            embedding_dim (int): Dimension of the input embeddings.
            hidden_dim (int): Dimension of the hidden state in the LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate for the LSTM layers.
        """
        super().__init__()
        
        self.lstm = LSTM(input_size=embedding_dim*2 + 6, # mu(t) + logvar(t) + act(t-1) + act(t) 
                         hidden_size=hidden_dim, 
                         num_layers=num_layers,
                         dropout=dropout, 
                         batch_first=True)
        self.batch_norm = BatchNorm1d(hidden_dim)
        self.fc_mu = Linear(hidden_dim, embedding_dim)
        self.fc_logvar = Linear(hidden_dim, embedding_dim)

    
    def predict(self, mu:torch.Tensor, logvar:torch.Tensor, act:torch.Tensor):
        """
        Predict future values using the LSTM model.
        Args:
            mu (torch.Tensor): Mean tensor of shape (batch_size, seq_len_obs, embedding_dim).
            logvar (torch.Tensor): Log variance tensor of shape (batch_size, seq_len_obs, embedding_dim).
            act (torch.Tensor): Action tensor of shape (batch_size, seq_len_pred, act_dim).
        Returns:
            torch.Tensor: Predicted mean tensor of shape (batch_size, seq_len_pred, embedding_dim).
        """
        batch_size, seq_len_pred, act_dim = act.shape
        _, seq_len_obs, embed_dim = mu.shape

        h_t, c_t = None, None  # For LSTM hidden state
        outputs_mu = []
        outputs_logvar = []

        # Start from t=0
        for t in range(seq_len_pred - 1):
            if t < seq_len_obs - 1:
                # Use ground truth
                mu_input = mu[:, t, :]
                logvar_input = logvar[:, t, :]
            else:
                # Use model prediction
                mu_input = outputs_mu[-1]
                logvar_input = outputs_logvar[-1]

            act_t = act[:, t, :]
            act_t1 = act[:, t + 1, :]

            lstm_input = torch.cat([mu_input, logvar_input, act_t, act_t1], dim=-1).unsqueeze(1)
            lstm_out, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t)) if h_t is not None else self.lstm(lstm_input)

            normed = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)
            mu_pred = self.fc_mu(normed)
            logvar_pred = self.fc_logvar(normed)

            outputs_mu.append(mu_pred.squeeze(1))
            outputs_logvar.append(logvar_pred.squeeze(1))

        return outputs_mu, outputs_logvar

            

    def forward_training_teacher_forcing(self, mu, logvar, act):
        """
        Forward pass for training.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
        Returns:
            float: The loss value.
        """
        # Concatenate mu, logvar, and act to form the input for LSTM
        x = torch.cat(
            (
                mu[:, :-1, :],
                logvar[:, :-1, :],
                act[:, :-1, :],
                act[:, 1:, :]
            ), 
            dim=-1
        )  # (batch_size, seq_length-1, embedding_dim*2 + act_dim)
        # convert to float32 if necessary
        x = x.float().contiguous()  # Ensure the tensor is contiguous in memory
        lstm_out, _ = self.lstm(x)
        lstm_out = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization
        mu_pred = self.fc_mu(lstm_out)
        mu_true = mu[:, 1:, :]
        logvar_pred = self.fc_logvar(lstm_out)
        logvar_true = logvar[:, 1:, :]
        mse_mu = mse_loss(mu_pred, mu_true)
        mse_logvar = mse_loss(logvar_pred, logvar_true)
        loss = mse_mu + mse_logvar
        return loss
    
    def forward_training_predicting(self, mu, logvar, act):
        """
        Forward pass for training without teacher forcing.
        Args:
            mu (torch.Tensor): Mean tensor of shape (batch_size, seq_len_obs, embedding_dim).
            logvar (torch.Tensor): Log variance tensor of shape (batch_size, seq_len_obs, embedding_dim).
            act (torch.Tensor): Action tensor of shape (batch_size, seq_len_pred, act_dim).
        Returns:
            float: The loss value.
        """
        mu = mu[:, 0:2, :] # Use the first frame of mu as the initial input
        logvar = logvar[:, 0:2, :] # Use the first frame of logvar as the initial input  
        mu = mu.float().contiguous()  # Ensure the tensor is contiguous in memory
        logvar = logvar.float().contiguous()  # Ensure the tensor is contiguous in memory
        act = act.float().contiguous()  # Ensure the tensor is contiguous in memory
        mu_pred, logvar_pred = self.predict(mu, logvar, act)
        mu_true = mu[:, -1, :]
        logvar_true = logvar[:, -1, :]
        mu_pred = mu_pred[-1]  # Get the last prediction
        logvar_pred = logvar_pred[-1]
        mse_mu = mse_loss(mu_pred, mu_true)
        mse_logvar = mse_loss(logvar_pred, logvar_true)
        loss = mse_mu + mse_logvar
        return loss
    
    def forward_training(self, mu, logvar, act, teacher_forcing:bool=True):
        """
        Forward pass for training.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            teacher_forcing (bool): Whether to use teacher forcing.
        Returns:
            float: The loss value.
        """
        if teacher_forcing:
            return self.forward_training_teacher_forcing(mu, logvar, act)
        else:
            return self.forward_training_predicting(mu, logvar, act) 


    def train_epoch(self, tr_loader, optimizer, device, teacher_forcing:bool=True):
        '''
        Train the model for one epoch.
        Args:
            tr_loader (DataLoader): DataLoader for the training set.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
            device (torch.device): Device to run the model on.
            teacher_forcing (bool): Whether to use teacher forcing.
        Returns:
            float: The average loss for the epoch.
        '''
        self.train()
        tot_loss = 0.0
        for mu, logvar, act in tr_loader:
            mu, logvar, act = mu.to(device), logvar.to(device), act.to(device)
            optimizer.zero_grad()
            loss = self.forward_training(mu, logvar, act, teacher_forcing=teacher_forcing)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * mu.size(0)
        return tot_loss / len(tr_loader.dataset)    

    def test_epoch(self, vs_loader, device, teacher_forcing:bool=True):
        """
        Test the model for one epoch.
        Args:
            vs_loader (DataLoader): DataLoader for the validation set.
            device (torch.device): Device to run the model on.
            teacher_forcing (bool): Whether to use teacher forcing.
        Returns:
            float: The average loss for the epoch.
        """
        self.eval()
        tot_loss = 0.0
        with torch.no_grad():
            for mu, logvar, act in vs_loader:
                mu, logvar, act = mu.to(device), logvar.to(device), act.to(device)
                loss = self.forward_training(mu, logvar, act, teacher_forcing=teacher_forcing)
                tot_loss += loss.item() * mu.size(0)
        return tot_loss / len(vs_loader.dataset)