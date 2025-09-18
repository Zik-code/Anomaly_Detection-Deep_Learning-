'''
Ya, S., Youjian, Z., Chenhao, N., Rong, L., Wei, S., & Dan, P. (2019) 
Robust Anomaly Detection for Multivariate Time Series Through Stochastic
 Recurrent Neural Network, ACM SIGKDD Conference on Knowledge Discovery  
 and Data Mining: 2828-2837.
'''


import torch
import torch.nn as nn
from src.constants import lr

class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is None else hidden
        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden