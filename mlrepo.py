import torch, numpy as np, random, optuna, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

class SoH_Est_LSTM(nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers, out_dim):
        super(SoH_Est_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(feat_dim, hidden_dim, n_layers, batch_first=True)
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()
        self.fc1 = nn.Linear(hidden_dim, 8)
        self.olayer = nn.Linear(8, out_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        out, _ = self.lstm(x, (h0, c0))
        out = self.olayer(self.act1(self.fc1(out[:, -1, :])))
        return self.act2(out.squeeze(-1))

class SeriesDataset(Dataset):
    """
    items: pandas dataframe
    """
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {'feat': self.items[idx]['feats'], 'target': self.items[idx]['targets']}

def collate_fn(batch):
    # Extract features and targets from the batch of dictionaries
    sequences = [torch.as_tensor(x['feat'], dtype=torch.float32) for x in batch]

    max_len = 61  # fixed max sequence length

    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        pad = torch.zeros(pad_len, seq.size(1))  # shape: (pad_len, 4)
        padded_seq = torch.cat([pad, seq], dim=0)
        padded_sequences.append(padded_seq)

    batch_tensor = torch.stack(padded_sequences)
    targets = torch.stack([torch.as_tensor(x['target'], dtype=torch.float32) for x in batch]).flatten()

    return batch_tensor, targets

