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
        
        self.embedding_dim = embedding_dim
        self.prep_fc = Linear(embedding_dim*2 + 6, embedding_dim*2 + 6)  # mu(t) + logvar(t) + act(t-1) + act(t)
        self.batch_norm_prep = BatchNorm1d(embedding_dim*2 + 6)
        self.prep_fc2 = Linear(embedding_dim*2 + 6, embedding_dim*2)
        self.batch_norm_prep2 = BatchNorm1d(embedding_dim*2)
        self.lstm = LSTM(input_size=embedding_dim*2,
                         hidden_size=hidden_dim, 
                         num_layers=num_layers,
                         dropout=dropout, 
                         batch_first=True)
        self.batch_norm_1 = BatchNorm1d(hidden_dim)
        self.fc_1 = Linear(hidden_dim + embedding_dim*2, hidden_dim)  # lstm_out + embedding_dim*2
        self.batch_norm_2 = BatchNorm1d(hidden_dim)
        self.fc_2 = Linear(hidden_dim, hidden_dim)  # lstm_out, (h_t, c_t)
        self.batch_norm_3 = BatchNorm1d(hidden_dim)
        self.fc_mu = Linear(hidden_dim, embedding_dim)
        self.fc_logvar = Linear(hidden_dim, embedding_dim)

    def forward(self, x, h_t=None, h_c=None):
        """
        Forward pass of the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, mu + logvar + act(t-1) + act(t)).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embedding_dim).
        """
        out = relu(self.prep_fc(x))  # Prepare the input
        out = self.batch_norm_prep(out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization
        out = relu(self.prep_fc2(out))  # Prepare the input
        skip = self.batch_norm_prep2(out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization
        out, (h_t, h_t) = self.lstm(skip) if h_t is None else self.lstm(skip, (h_t, h_t))
        out = self.batch_norm_1(out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization
        out = torch.cat((out, skip), dim=-1)  # Concatenate the skip connection
        out = relu(self.fc_1(out))
        out = self.batch_norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = relu(self.fc_2(out))
        out = self.batch_norm_3(out.transpose(1, 2)).transpose(1, 2)
        mu = self.fc_mu(out) #+ x[:, :, :self.embedding_dim]
        logvar = self.fc_logvar(out) #+ x[:, :, self.embedding_dim:2*self.embedding_dim]
        return mu, logvar, (h_t, h_t)
        

    
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
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
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
                    mu_input = outputs_mu[-1] #+ mu_input
                    logvar_input = outputs_logvar[-1] #+ logvar_input

                act_t = act[:, t, :]
                act_t1 = act[:, t + 1, :]

                lstm_input = torch.cat([mu_input, logvar_input, act_t, act_t1], dim=-1).unsqueeze(1)
                mu_pred, logvar_pred, (h_t, c_t) = self.forward(lstm_input, h_t, c_t)
                outputs_mu.append(mu_pred.squeeze(1))
                outputs_logvar.append(logvar_pred.squeeze(1))

            return outputs_mu, outputs_logvar

            

    def forward_teacher_forcing(self, mu, logvar, act, device:torch.device='cpu'):
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
        x = x.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        mu = mu.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        logvar = logvar.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        act = act.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        mu_pred, logvar_pred, _ = self.forward(x)  # Forward pass through the LSTM
        mse_mu = mse_loss(mu_pred, mu[:, 1:, :].to(device))  # Compare with the true mu
        mse_logvar = mse_loss(logvar_pred, logvar[:, 1:, :].to(device))  # Compare with the true logvar
        loss = mse_mu + mse_logvar + self.kl_divergence(mu_pred, logvar_pred)  # Add KL divergence term
        return loss
    
    def forward_predicting(self, mu, logvar, act, device:torch.device='cpu'):
        """
        Forward pass for training without teacher forcing.
        Args:
            mu (torch.Tensor): Mean tensor of shape (batch_size, seq_len_obs, embedding_dim).
            logvar (torch.Tensor): Log variance tensor of shape (batch_size, seq_len_obs, embedding_dim).
            act (torch.Tensor): Action tensor of shape (batch_size, seq_len_pred, act_dim).
        Returns:
            float: The loss value.
        """
        mu = mu.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        logvar = logvar.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        act = act.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        mu_true = mu[:, 1:, :] #- mu[:, :-1, :] # compute target mu
        logvar_true = logvar[:, 1:, :] #- logvar[:, :-1, :] # compute target logvar

        mu_1 = mu[:, 0, :]  # Get the first mu
        logvar_1 = logvar[:, 0, :]  # Get the first logvar
        mu_pred, logvar_pred = [], []
        h_t, c_t = None, None  # For LSTM hidden state
        for t in range(act.shape[1] - 1):
            act_t = act[:, t, :]
            act_t1 = act[:, t + 1, :]
            #print(f"t: {t}, act_t: {act_t.shape}, act_t1: {act_t1.shape}, mu_1: {mu_1.shape}, logvar_1: {logvar_1.shape}")
            lstm_input = torch.cat([mu_1, logvar_1, act_t, act_t1], dim=-1).unsqueeze(1)  # (batch_size, 1, embedding_dim*2 + act_dim)
            mu_1, logvar_1, (h_t, c_t) = self.forward(lstm_input, h_t, c_t)  # Forward pass through the LSTM
            mu_1 = mu_1.squeeze(1)  # Remove the sequence dimension
            logvar_1 = logvar_1.squeeze(1)  # Remove the sequence dimension
            mu_pred.append(mu_1)  # Append the predicted mu
            logvar_pred.append(logvar_1)  # Append the predicted logvar
        mu_pred = torch.stack(mu_pred, dim=1)  # (batch_size, seq_len_pred-1, embedding_dim)
        logvar_pred = torch.stack(logvar_pred, dim=1)  # (batch_size, seq_len_pred-1, embedding_dim)
        mse_mu = mse_loss(mu_pred, mu_true)  # Compare with the true mu
        mse_logvar = mse_loss(logvar_pred, logvar_true)  # Compare with the true logvar
        loss = mse_mu + mse_logvar + self.kl_divergence(mu_pred, logvar_pred)  # Add KL divergence term
        return loss
    
    def forward_light_predicting(self, mu, logvar, act, device:torch.device='cpu'):
        """
        Forward pass for training without teacher forcing.
        Args:
            mu (torch.Tensor): Mean tensor of shape (batch_size, seq_len_obs, embedding_dim).
            logvar (torch.Tensor): Log variance tensor of shape (batch_size, seq_len_obs, embedding_dim).
            act (torch.Tensor): Action tensor of shape (batch_size, seq_len_pred, act_dim).
        Returns:
            float: The loss value.
        """
        mu = mu.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        logvar = logvar.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory
        act = act.float().contiguous().to(device)  # Ensure the tensor is contiguous in memory

        mu_target_last_step = mu[:, -1, :]
        logvar_target_last_step = logvar[:, -1, :]
        current_mu = mu[:, 0, :]
        current_logvar = logvar[:, 0, :]
        
        h_t, c_t = None, None  # Initialize LSTM hidden and cell states

        # Iterate through the sequence of actions to make predictions step-by-step
        # The loop runs act.shape[1] - 1 times.
        # After the loop, current_mu and current_logvar will hold the prediction for the final step.
        for t in range(act.shape[1] - 1):
            act_t = act[:, t, :]
            act_t1 = act[:, t + 1, :]
            
            lstm_input = torch.cat([current_mu, current_logvar, act_t, act_t1], dim=-1).unsqueeze(1)
            predicted_mu_step, predicted_logvar_step, (h_t, c_t) = self.forward(lstm_input, h_t, c_t) if h_t is not None else self.forward(lstm_input)
            current_mu = predicted_mu_step.squeeze(1)
            current_logvar = predicted_logvar_step.squeeze(1)

        mse_mu = mse_loss(current_mu, mu_target_last_step)
        mse_logvar = mse_loss(current_logvar, logvar_target_last_step)
        loss = mse_mu + mse_logvar + self.kl_divergence(current_mu, current_logvar)  # Add KL divergence term
        return loss
    
    def kl_divergence(self, mu:torch.Tensor, logvar:torch.Tensor):
        """
        Compute the KL divergence between the predicted and true distributions.
        Args:
            mu (torch.Tensor): Mean tensor of shape (batch_size, seq_len, embedding_dim).
            logvar (torch.Tensor): Log variance tensor of shape (batch_size, seq_len, embedding_dim).
        Returns:
            torch.Tensor: KL divergence value.
        """
        # new shape is (batch_size* seq_len, embedding_dim)
        mu = mu.view(-1, mu.size(-1))  # Flatten the tensor
        logvar = logvar.view(-1, logvar.size(-1))
        # Compute KL divergence
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        return kld.mean()*0
    
    def forward_wrapper(self, mu, logvar, act, teacher_forcing:bool=True, full_error:bool=False, device:torch.device='cpu'):
        """
        Forward pass for training.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            teacher_forcing (bool): Whether to use teacher forcing.
        Returns:
            float: The loss value.
        """
        if teacher_forcing:
            if full_error:
                raise ValueError("Teacher forcing with full error is not supported.")
            return self.forward_teacher_forcing(mu, logvar, act, device=device)
        else:
            if full_error:
                return self.forward_predicting(mu, logvar, act, device=device)
            else:
                return self.forward_light_predicting(mu, logvar, act, device=device) 


    def train_epoch(self, tr_loader, optimizer, device, teacher_forcing:bool=True, full_error:bool=False):
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
            optimizer.zero_grad()
            loss = self.forward_wrapper(mu, logvar, act, teacher_forcing=teacher_forcing, device=device, full_error=full_error)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * mu.size(0)
        return tot_loss / len(tr_loader.dataset)    

    def test_epoch(self, vs_loader, device, teacher_forcing:bool=True, full_error:bool=False):
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
                loss = self.forward_wrapper(mu, logvar, act, teacher_forcing=teacher_forcing, device=device, full_error=full_error)
                tot_loss += loss.item() * mu.size(0)
        return tot_loss / len(vs_loader.dataset)