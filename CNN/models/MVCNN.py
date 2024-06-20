import torch
import torch.nn as nn
from models.mlp_head import MLPHead

# MVCNN 
class MVCNN(torch.nn.Module):
    def __init__(self, config,device):
        super(MVCNN, self).__init__()
        self.device = device
        self.num_view = config.num_view
        if config.name == 'CNN3':
            self.net = CNN3()
        elif config.name == 'CNN4':
            self.net = CNN4()
        elif config.name == 'CNN5':
            self.net = CNN5()
        elif config.name == 'CNN6':
            self.net = CNN6()
        self.feature_dim = 64*12*24
                   
        if config.cls==1:
            self.fc = nn.Linear(self.feature_dim, config.projection_head)
        else:
            self.fc = MLPHead(in_channels=self.feature_dim, projection_size=config.projection_head)

    def forward(self, x, view=False):
        view_pool = []
        for i in range(x.size(1)):
            x_ = x[:,i,:,:,:] # (batch_size, view_num, ch, w, h)
            h = self.net(x_) # 各誘導を入力
            feature_map_size = h.size()
            h = h.view(x.size(0),-1)
            view_pool.append(h)
        
        # max pool for all view each channel
        pooled_view = view_pool[0]
        for i in range(1,len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])
        
        if view:
            return torch.reshape(pooled_view,feature_map_size)
        
        out = self.fc(pooled_view)
        return out
    
# MVCNN CNN複数版
class MVCNN_Net(torch.nn.Module):
    def __init__(self, config,device):
        super(MVCNN_Net, self).__init__()
        self.device = device
        self.num_view = config.num_view
        if config.name == 'CNN3':
            self.net = torch.nn.ModuleList(CNN3() for i in range(config.num_view))
        elif config.name == 'CNN4':
            self.net = torch.nn.ModuleList(CNN4() for i in range(config.num_view))
        elif config.name == 'CNN5':
            self.net = torch.nn.ModuleList(CNN5() for i in range(config.num_view))
        elif config.name == 'CNN6':
            self.net = torch.nn.ModuleList(CNN6() for i in range(config.num_view))
            
        self.feature_dim = 32*12*24
        
        if config.cls==1:
            self.fc = nn.Linear(self.feature_dim, config.projection_head)
        else:
            self.fc = MLPHead(in_channels=self.feature_dim, projection_size=config.projection_head)

    def forward(self, x):
        view_pool = []
        for i ,net in enumerate(self.net):
            x_ = x[:,i,:,:,:] # (batch_size, view_num, ch, w, h)
            h = net(x_) # 各誘導を入力
            h = h.view(x.size(0),-1)
            view_pool.append(h)
        
        # max pool for all view each channel
        pooled_view = view_pool[0]
        for i in range(1,len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])
        
        out = self.fc(pooled_view)
        return out


class CNN3(torch.nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class CNN4(torch.nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class CNN5(torch.nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class CNN6(torch.nn.Module):
    def __init__(self):
        super(CNN6, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x    
    
