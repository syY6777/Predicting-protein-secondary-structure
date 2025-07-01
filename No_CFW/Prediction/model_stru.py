import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=50,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.PReLU(),
            nn.BatchNorm1d(50),

        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 100, 3, 2, 0),
            nn.PReLU(),
            nn.BatchNorm1d(100),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(100, 50, 3, 2, 1),
            nn.PReLU(),
            nn.BatchNorm1d(50),
        )

        self.attention3 = nn.MultiheadAttention(embed_dim=45, num_heads=5)
        self.batchnorm3 = nn.BatchNorm1d(45)
        self.conv4 = nn.Sequential(
            nn.Conv1d(50, 3, 3, 1, 0),
            nn.PReLU(),
            nn.BatchNorm1d(3),
        )

        self.l1 = nn.Linear(30, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        out = F.relu(self.l1(x))
        return out
