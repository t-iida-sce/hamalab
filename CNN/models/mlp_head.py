from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels,projection_size,mid_channels=False):
        super(MLPHead, self).__init__()
        if not mid_channels:
            mid_channels =in_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, projection_size)
        )

    def forward(self, x):
        return self.net(x)
