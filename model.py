import torch
import torch.nn as nn
import torch.nn.functional as F


class LBR(nn.Module):
    def __init__(self, d_model, bias):
        super(LBR, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(d_model, d_model, bias),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.main(x)
        return x


class LBRD(nn.Module):
    def __init__(self, d_model, bias, dropout):
        super(LBRD, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(d_model, d_model, bias),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.main(x)
        return x


