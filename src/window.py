import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NeuralSequenceDataset(Dataset):
    '''
    We create a sliding-window of neural responses for LSTM decoding.
    '''

    def __init__(self, X, y, seq_len = 50, predict_ahead = 0):

        # take data and convert into a tensor and store as attribute
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

        # initialize the sequence length (also we only predict one time point ahead)
        self.seq_len = seq_len
        self.predict_ahead = predict_ahead

        # valid length of how we slide
        self.valid_length = len(self.X) - seq_len - predict_ahead

    def __len__(self):
        return self.valid_length
    
    def __getitem__(self, idx):
        start= idx
        stop = idx + self.seq_len

        x_seq = self.X[start : stop]
        y_target = self.y[stop + self.predict_ahead - 1]


