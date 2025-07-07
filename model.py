# model.py

import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super(MusicLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # shape: [batch_size, seq_len, embed_dim]
        out, hidden = self.lstm(x, hidden)  # out: [batch_size, seq_len, hidden_dim]
        out = self.fc(out[:, -1, :])  # use only the last time step's output
        return out, hidden
