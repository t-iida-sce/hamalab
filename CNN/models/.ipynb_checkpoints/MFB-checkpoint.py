import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp_head import MLPHead

# Multi future branch neural network
# MVCNN CNN複数版
class MFB_one(torch.nn.Module):
    def __init__(self, config,device):
        super(MFB_one, self).__init__()
        self.device = device
        self.num_view = config.num_view
        if config.name == 'CNN2':
            self.net = CNN2()
        elif config.name == 'CNN3':
            self.net = CNN3()
        elif config.name == 'CNN4':
            self.net = CNN4()
        
        self.densenet = DenseNet(num_view=config.num_view, growthRate=24, reduction=0.2, bottleneck=True)
        self.feature_dim = 312
        if config.cls==1:
            self.fc = nn.Linear(self.feature_dim, config.projection_head)
        else:
            self.fc = MLPHead(in_channels=self.feature_dim, projection_size=config.projection_head)

    def forward(self, x):
        views = []
        for i in range(x.size(1)):
            x_ = x[:,i,:,:,:] # (batch_size, view_num, ch, w, h)
            h = self.net(x_) # 各誘導を入力
            views.append(h)
        views = torch.stack(views)
        views =  torch.permute(views, (1,0,2,3,4))
        views = views.reshape(views.size(0),-1,views.size(3),views.size(4))
        features = self.densenet(views).squeeze(2).squeeze(2)
        out = self.fc(features)
        return out

class MFB(torch.nn.Module):
    def __init__(self, config,device):
        super(MFB, self).__init__()
        self.device = device
        self.num_view = config.num_view
        if config.name == 'CNN2':
            self.net = torch.nn.ModuleList(CNN2() for i in range(config.num_view))
        elif config.name == 'CNN3':
            self.net = torch.nn.ModuleList(CNN3() for i in range(config.num_view))
        elif config.name == 'CNN4':
            self.net = torch.nn.ModuleList(CNN4() for i in range(config.num_view))
        
        self.densenet = DenseNet(num_view=config.num_view, growthRate=24, reduction=0.2, bottleneck=True)
        self.feature_dim = 312
        if config.cls==1:
            self.fc = nn.Linear(self.feature_dim, config.projection_head)
        else:
            self.fc = MLPHead(in_channels=self.feature_dim, projection_size=config.projection_head)

    def forward(self, x):
        views = []
        for i ,net in enumerate(self.net):
            x_ = x[:,i,:,:,:] # (batch_size, view_num, ch, w, h)
            h = net(x_) # 各誘導を入力
            views.append(h)
        views = torch.stack(views)
        views =  torch.permute(views, (1,0,2,3,4))
        views = views.reshape(views.size(0),-1,views.size(3),views.size(4))
        features = self.densenet(views).squeeze(2).squeeze(2)
        out = self.fc(features)
        return out

class CNN2(torch.nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class CNN3(torch.nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())

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
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
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
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x    
    
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, dropout=0.2):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1, bias=False)

        self.drop_rate = dropout
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, dropout=0.2):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.drop_rate = dropout
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_view, growthRate, reduction, bottleneck=True):
        super(DenseNet, self).__init__()

        nChannels = 48
        self.conv1 = nn.Conv2d(32*num_view, nChannels, kernel_size=7, padding=3,bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, 6, bottleneck)
        nChannels += 6*growthRate
        nOutChannels = 24
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, 24, bottleneck)
        nChannels += 24*growthRate
        nOutChannels = 24
        self.trans2 = Transition(nChannels, nOutChannels)

        #nChannels = nOutChannels
        #self.dense3 = self._make_dense(nChannels, growthRate, 32, bottleneck)
        #nChannels += 32*growthRate
        #nOutChannels = 24
        #self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, 12, bottleneck)
        nChannels += 12*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        #out = self.dense3(out)
        #out = self.trans3(out)
        out = self.dense4(out)
        out = self.avgpool(F.relu(self.bn1(out)))
        return out