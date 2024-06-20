import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import models.resnet
from omegaconf import OmegaConf

# compornents
class conv2d_layer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(conv2d_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)
    
class convTrans2d_layer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,last_layer=False):
        super(convTrans2d_layer, self).__init__()
        if not last_layer:
            self.convTrans = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels, momentum=0.01),
                nn.ReLU(inplace=True),
            )
        else:
            self.convTrans = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels, momentum=0.01),
                nn.Sigmoid(),
            )
    def forward(self,x):
        return self.convTrans(x)
    
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
## ---------------------- (Normal) AutoEncoder ---------------------- ##    

class encoder_ae(nn.Module):
    def __init__(self,inp_chs,out_chs,kernels,strides,paddings,admaxpool_h,admaxpool_w,neurons_middle_layers,CNN_embed_dim,kd=False):
        super(encoder_ae,self).__init__()
        self.encode_list = nn.ModuleList([conv2d_layer(i_ch,o_ch,k,s,p) for i_ch,o_ch,k,s,p in zip(inp_chs,out_chs,kernels,strides,paddings)])
        self.encode_list.extend([nn.AdaptiveMaxPool2d((admaxpool_h,admaxpool_w)),nn.Flatten(),nn.Linear(neurons_middle_layers, CNN_embed_dim)])
    def forward(self,x,kd=False):
        for enc in self.encode_list:
            x = enc(x)
        return x
    
class decoder_ae(nn.Module):
    def __init__(self,batch_size,input_size,inp_chs,out_chs,kernels,strides,paddings,admaxpool_h,admaxpool_w,neurons_middle_layers,CNN_embed_dim):
        super(decoder_ae,self).__init__()
        self.input_size = input_size
        self.out_chs = out_chs
        self.admaxpool_h = admaxpool_h
        self.admaxpool_w = admaxpool_w
        self.decode_fc = nn.Sequential(
            nn.Linear(CNN_embed_dim, neurons_middle_layers),
            nn.ReLU(inplace=True),
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        deconv_list = [convTrans2d_layer(o_ch,i_ch,k,s,p,True) if not i else convTrans2d_layer(o_ch,i_ch,k,s,p,False) for i,(i_ch,o_ch,k,s,p) in enumerate(zip(inp_chs,out_chs,kernels,strides,paddings))]
        self.decode_list = nn.ModuleList(deconv_list[::-1])

    def forward(self,x):
        x = self.decode_fc(x)
        x = x.view(-1,self.out_chs[-1],self.admaxpool_h,self.admaxpool_w)
        x = self.upsampling(x)
        for dec in self.decode_list:
            x = dec(x)
        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        return x
    
class AE(nn.Module):
    def __init__(self, batch_size,input_ch,CNN_embed_dim,input_size):
        super(AE, self).__init__()

        self.batch_size = batch_size
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.input_ch = input_ch
        self.input_size = input_size
        self.inp_chs = [self.input_ch,16, 32, 64]
        self.out_chs = [16, 32, 64, 128]
        self.kernels  = [(3, 3), (3, 3), (3, 3), (2, 2)]      # 2d kernal size
        self.strides  = [(3, 3), (2, 2), (2, 2), (2, 2)]# 2d strides
        self.paddings = [(0, 0), (0, 0), (0, 0), (0, 0)] # 2d padding
        self.admaxpool_h = 2
        self.admaxpool_w = 6
        self.neurons_middle_layers = self.out_chs[-1] * self.admaxpool_h *self.admaxpool_w # for avepooling

        # encoding
        self.encoder = encoder_ae(self.inp_chs,self.out_chs,self.kernels,self.strides,self.paddings,self.admaxpool_h,self.admaxpool_w,self.neurons_middle_layers,self.CNN_embed_dim)

        # Decoding
        self.decoder = decoder_ae(self.batch_size,self.input_size,self.inp_chs,self.out_chs,self.kernels,self.strides,self.paddings,self.admaxpool_h,self.admaxpool_w,self.neurons_middle_layers,self.CNN_embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconst = self.decoder(z)
        return x_reconst, z
    
    
## ---------------------- (Normal) VAE ---------------------- ##
class encoder_vae(nn.Module):
    def __init__(self,inp_chs,out_chs,kernels,strides,paddings,admaxpool_h,admaxpool_w,neurons_middle_layers,CNN_embed_dim):
        super(encoder_vae,self).__init__()
        self.encode_list = nn.ModuleList([conv2d_layer(i_ch,o_ch,k,s,p) for i_ch,o_ch,k,s,p in zip(inp_chs,out_chs,kernels,strides,paddings)])
        self.encode_list.extend([nn.AdaptiveMaxPool2d((admaxpool_h,admaxpool_w)),nn.Flatten()])
        
        # Latent vectors mu and sigma
        self.fc1_mu = nn.Linear(neurons_middle_layers, CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc1_logvar = nn.Linear(neurons_middle_layers, CNN_embed_dim)  # output = CNN embedding latent variables
        
    def forward(self,x,kd=False):
        for enc in self.encode_list:
            x = enc(x)
        mu, logvar = self.fc1_mu(x), self.fc1_logvar(x)
        if kd:
            return mu
        return mu, logvar

class decoder_vae(nn.Module):
    def __init__(self,batch_size,input_size,inp_chs,out_chs,kernels,strides,paddings,admaxpool_h,admaxpool_w,neurons_middle_layers,CNN_embed_dim):
        super(decoder_vae,self).__init__()
        self.input_size = input_size
        self.out_chs = out_chs
        self.admaxpool_h = admaxpool_h
        self.admaxpool_w = admaxpool_w
        self.decode_fc = nn.Sequential(
            nn.Linear(CNN_embed_dim, neurons_middle_layers),
            nn.ReLU(inplace=True),
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        deconv_list = [convTrans2d_layer(o_ch,i_ch,k,s,p,True) if not i else convTrans2d_layer(o_ch,i_ch,k,s,p,False) for i,(i_ch,o_ch,k,s,p) in enumerate(zip(inp_chs,out_chs,kernels,strides,paddings))]
        self.decode_list = nn.ModuleList(deconv_list[::-1])

    def forward(self,x):
        x = self.decode_fc(x)
        x = x.view(-1,self.out_chs[-1],self.admaxpool_h,self.admaxpool_w)
        x = self.upsampling(x)
        for dec in self.decode_list:
            x = dec(x)
        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        return x
    
class VAE(nn.Module):
    def __init__(self, batch_size,input_ch,CNN_embed_dim,input_size):
        super(VAE, self).__init__()

        self.batch_size = batch_size
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.input_ch = input_ch
        self.input_size = input_size
        self.inp_chs = [self.input_ch,16, 32, 64]
        self.out_chs = [16, 32, 64, 128]
        self.kernels  = [(3, 3), (3, 3), (3, 3), (2, 2)]      # 2d kernal size
        self.strides  = [(3, 3), (2, 2), (2, 2), (2, 2)]# 2d strides
        self.paddings = [(0, 0), (0, 0), (0, 0), (0, 0)] # 2d padding
        self.admaxpool_h = 2
        self.admaxpool_w = 6
        self.neurons_middle_layers = self.out_chs[-1] * self.admaxpool_h *self.admaxpool_w # for avepooling

        # encoding
        self.encoder = encoder_vae(self.inp_chs,self.out_chs,self.kernels,self.strides,self.paddings,self.admaxpool_h,self.admaxpool_w,self.neurons_middle_layers,self.CNN_embed_dim)

        # Decoding
        self.decoder = decoder_vae(self.batch_size,self.input_size,self.inp_chs,self.out_chs,self.kernels,self.strides,self.paddings,self.admaxpool_h,self.admaxpool_w,self.neurons_middle_layers,self.CNN_embed_dim)
       

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decoder(z)
        return x_reconst, z, mu, logvar
    
## ---------------------- ResNet AE ---------------------- ##

class ResNet_AE(nn.Module):
    def __init__(self, input_ch,CNN_embed_dim,input_size,s=1):
        super(ResNet_AE, self).__init__()

        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.input_ch = input_ch
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (4, 4), (4, 4), (2, 2), (2, 2)     # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (4, 4), (4, 4), (2, 2), (1, 1)    # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.input_size=OmegaConf.to_container(input_size)
        self.admaxpool_h = 1
        self.admaxpool_w = 1
        self.s = s
        self.neurons_middle_layers = self.ch4*s * self.admaxpool_h *self.admaxpool_w # for avepooling

        # encoding components
        if self.s == 1:
            resnet = resnet_attention.resnet18s(self.input_ch,pretrained=False)
        elif self.s==2:
            resnet = resnet_attention.resnet18(self.input_ch,pretrained=False)
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.admaxpool = nn.AdaptiveMaxPool2d((self.admaxpool_h,self.admaxpool_w))

        # Latent vectors 
        self.fc1 = nn.Linear(self.neurons_middle_layers*self.s, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc2 = nn.Sequential(
            nn.Linear(self.CNN_embed_dim, self.neurons_middle_layers*3),
            nn.ReLU(inplace=True)
        ) 


        # Decoder
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch4, out_channels=self.ch3, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch3, out_channels=self.ch2, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch2, out_channels=self.ch1, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch1, out_channels=self.input_ch, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1),
            nn.BatchNorm2d(self.input_ch, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = self.admaxpool(x)
        x = x.view(-1, self.neurons_middle_layers*self.s) # flatten output of conv
        # FC layers
        x = self.fc1(x)
        return x


    def decode(self, z):
        x = self.fc2(z).view(-1,self.ch4,  self.admaxpool_h,  self.admaxpool_w*3)
        x = self.upsampling(x)
        x = self.convTrans5(x)      
        x = self.convTrans6(x)    
        x = self.convTrans7(x)     
        x = self.convTrans8(x)
        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)
        return x_reconst, z
    
## ---------------------- ResNet VAE ---------------------- ##

class ResNet_VAE(nn.Module):
    def __init__(self, input_ch,CNN_embed_dim,input_size, s=1):
        super(ResNet_VAE, self).__init__()

        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.input_ch = input_ch
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (4, 4), (4, 4), (2, 2), (2, 2)     # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (4, 4), (4, 4), (2, 2), (1, 1)    # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.input_size=input_size
        self.admaxpool_h = 1
        self.admaxpool_w = 1
        self.s = s
        self.neurons_middle_layers = self.ch4 * self.admaxpool_h *self.admaxpool_w # for avepooling

        # encoding components
        if self.s == 1:
            resnet = resnet_attention.resnet18s(self.input_ch,pretrained=False)
        elif self.s==2:
            resnet = resnet_attention.resnet18(self.input_ch,pretrained=False)
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.admaxpool = nn.AdaptiveMaxPool2d((self.admaxpool_h,self.admaxpool_w))

        # Latent vectors mu and sigma
        self.fc1_mu = nn.Linear(self.neurons_middle_layers*self.s, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc1_logvar = nn.Linear(self.neurons_middle_layers*self.s, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc3 = nn.Linear(self.CNN_embed_dim, self.neurons_middle_layers*2)

        # Decoder
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch4, out_channels=self.ch3, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch3, out_channels=self.ch2, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch2, out_channels=self.ch1, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch1, out_channels=self.input_ch, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1),
            nn.BatchNorm2d(self.input_ch, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = self.admaxpool(x)
        x = x.view(-1, self.neurons_middle_layers*self.s) # flatten output of conv, Adjusted by self.s
        # FC layers
        mu, logvar = self.fc1_mu(x), self.fc1_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.fc3(z).view(-1,self.ch4,self.admaxpool_h, self.admaxpool_w*2)
        #x = self.upsample(x)
        x = self.convTrans5(x)      
        x = self.convTrans6(x)    
        x = self.convTrans7(x)     
        x = self.convTrans8(x)
        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return x_reconst, z, mu, logvar