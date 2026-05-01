# neural network imports
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np

# import the src folder
import os
import sys
sys.path.append(os.path.abspath("../src"))

# import the sliding window creation function 
import window

class LSTMDecoder(nn.Module):
    '''
    LSTM decoder for predicting continuous finger position
    '''

    def __init__(self, input_dim = 96, hidden_dim = 64, num_layers = 1, dropout = 0.0):

        super().__init__()

        '''
        Create the LSTM
            input_size: 96 because of the number of channels
            hidden_dim: the more hidden dimensions the better the model captures complex patterns
            num_layers: 1 is enough for neural tasks
            dropout: we set 0 dropout here because only 1 layer
        
        Output:
            sequence of hidden states, each of size 64
        '''
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        # separate output layer after the 
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        '''
        x shape: (batch, seq_len, input_dim)
        '''

        lstm_out, _ = self.lstm(x)

        # take the final time step as output
        last_hidden = lstm_out[:, -1, :]

        y_pred = self.output_layer(last_hidden)

        return y_pred.squeeze(-1)


    def time_split(X, y, train_frac=0.8):
        '''
        Splitting the training data, default split is 80%
        '''
        n = len(X)
        split_idx = int(train_frac * n)

        X_train = X[:split_idx]
        y_train = y[:split_idx]
            
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test

