from torch import nn
import torch

class MLP_embbeding(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size,emmbeding_size, finetuning=0):
        super(MLP_embbeding, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, emmbeding_size),
        )

    def forward(self, x):
        return self.net(x)

class MLP_Classification(nn.Module):
    def __init__(self,inchanels,classification_class):
        super(MLP_Classification2,self).__init__()
        self.fc1 = nn.Linear(inchanels,int(inchanels/2))
        self.fc2 = nn.Linear(int(inchanels/2),classification_class)

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MLP_Classification2(nn.Module):
    def __init__(self,inchanels,classification_class):
        super(MLP_Classification3,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inchanels,int(inchanels/2)),
            nn.BatchNorm1d(int(inchanels/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(inchanels/2),int(inchanels/4)),
            nn.BatchNorm1d(int(inchanels/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(inchanels/4),classification_class)
        )

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.net(x)
        return x

class Pool_MLPHead(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=False):
        super(PoolMLPHead, self).__init__()
        if not mid_channels:
            mid_channels = in_channels
        self.
        self.net = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels)
        )

    def forward(self, x):
        pool_view = torch.mean(x,1)
        out = self.net(pool_view)
        return out
