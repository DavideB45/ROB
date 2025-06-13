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

    def forward(self, x):
        """
        Forward pass of the MuLogvarLSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
        Returns:
            tuple: A tuple containing the mean and log variance tensors.
        """
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_hidden = lstm_out[:, -1, :]
        # Apply batch normalization
        last_hidden = self.batch_norm(last_hidden)
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return mu, logvar
    
    def predict(self, x, steps):
        """
        Predict future values using the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            steps (int): Number of future steps to predict.
        Returns:
            torch.Tensor: Predicted values of shape (batch_size, steps, embedding_dim).
        """
        lstm_out, _ = self.lstm(x)
        predictions = []
        for _ in range(steps):
            last_hidden = lstm_out[:, -1, :]
            last_hidden = self.batch_norm(last_hidden)
            mu = self.fc_mu(last_hidden)
            logvar = self.fc_logvar(last_hidden)
            predictions.append(mu.unsqueeze(1))
        predictions = torch.cat(predictions, dim=1)
        return predictions
    
    def forward_training_teacher_forcing(self, x):
        """
        Forward pass for training.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
        Returns:
            float: The loss value.
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.batch_norm(lstm_out)
        mu_pred = self.fc_mu(lstm_out)
        mu_pred = mu_pred[:, :-1, :]
        mu_true = x[:, 1:, :self.fc_logvar.out_features]
        logvar_pred = self.fc_logvar(lstm_out)
        logvar_pred = logvar_pred[:, :-1, :]
        logvar_true = x[:, 1:, self.fc_logvar.out_features:-6]
        mse_mu = mse_loss(mu_pred, mu_true)
        mse_logvar = mse_loss(logvar_pred, logvar_true)
        loss = mse_mu + mse_logvar
        return loss
    
    def forward_training(self, x, teacher_forcing:bool=False):
        """
        Forward pass for training.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            teacher_forcing (bool): Whether to use teacher forcing.
        Returns:
            float: The loss value.
        """
        if teacher_forcing:
            return self.forward_training_teacher_forcing(x)
        else:
            raise NotImplementedError("Teacher forcing is required for training.") 


