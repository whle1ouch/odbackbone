import torch
import torch.nn as nn
from torchsummary import summary


class ConvBN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
class Inception_A(nn.Module):
    
    def __init__(self, in_channels, channels_config):
        super(Inception_A, self).__init__()
        self.add_module("p1", ConvBN(in_channels, channels_config[0], 1))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, channels_config[1], 1),
            ConvBN(channels_config[1], channels_config[2], 3, padding=1)
        ))
        self.add_module("p3", nn.Sequential(
            ConvBN(in_channels, channels_config[3], 1),
            ConvBN(channels_config[3], channels_config[4], 3, padding=1),
            ConvBN(channels_config[4], channels_config[4], 3, padding=1)
        ))
        self.add_module("p4", nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBN(in_channels, channels_config[5], 1)
        ))
        
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x),
            self.p4(x)
        ], 1)
        

class Inception_B(nn.Module):
    
    def __init__(self, in_channels, channels_config, n=7):
        super(Inception_B, self).__init__()
        self.add_module("p1", ConvBN(in_channels, channels_config[0], 1))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, channels_config[1], 1),
            ConvBN(channels_config[1], channels_config[1], (1, 3), padding=(0, 1)),
            ConvBN(channels_config[1], channels_config[2], (3, 1), padding=(1, 0))
        ))
        self.add_module("p3", nn.Sequential(
            ConvBN(in_channels, channels_config[3], 1),
            ConvBN(channels_config[3], channels_config[3], (n, 1), padding=((n - 1) // 2, 0)),
            ConvBN(channels_config[3], channels_config[3], (1, n), padding=(0, (n - 1) // 2)),
            ConvBN(channels_config[3], channels_config[3], (n, 1), padding=((n - 1) // 2, 0)),
            ConvBN(channels_config[3], channels_config[4], (1, n), padding=(0, (n - 1) // 2)),
        )) 
        self.add_module("p4", nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBN(in_channels, channels_config[5], 1)
        ))
    
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x),
            self.p4(x)
        ], 1)
        

class Inception_C(nn.Module):
    
    def __init__(self, in_channels, channels_config):
        super(Inception_C, self).__init__()
        self.add_module("p1", ConvBN(in_channels, channels_config[0], 1))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, channels_config[1], 1),
            ConvBN(channels_config[1], channels_config[2], (1, 3), padding=(0, 1)),
            ConvBN(channels_config[2], channels_config[2], (3, 1), padding=(1, 0))
        ))
        self.add_module("p3", nn.Sequential(
            ConvBN(in_channels, channels_config[3], 1),
            ConvBN(channels_config[3], channels_config[4], 3, padding=1),
            ConvBN(channels_config[4], channels_config[4], (1, 3), padding=(0, 1)),
            ConvBN(channels_config[4], channels_config[4], (3, 1), padding=(1, 0)),
        )) 
        self.add_module("p4", nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBN(in_channels, channels_config[5], 1)
        ))
        
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x),
            self.p4(x)
        ], 1)
        

class Reduction_A(nn.Module):

    def __init__(self, in_channels, channels_config):
        super(Reduction_A, self).__init__()
        self.add_module("p1", ConvBN(in_channels, channels_config[0], 3, 2, 0))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, channels_config[1], 3, 1, 1),
            ConvBN(channels_config[1], channels_config[2], 3, 1, 1),
            ConvBN(channels_config[2], channels_config[2], 3, 2, 0)
        ))
        self.add_module("p3", nn.MaxPool2d(3, 2, 0))
        
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x)
        ], 1)
        

class Reduction_B(nn.Module):
    
    def __init__(self, in_channels, channels_config, n=7):
        super(Reduction_B, self).__init__()
        self.add_module("p1", nn.Sequential(
            ConvBN(in_channels, channels_config[0], 3, 1, 1),
            ConvBN(channels_config[0], channels_config[1], 3, 2, 0),
        ))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, channels_config[2], 3, 1, 1),
            ConvBN(channels_config[2], channels_config[2], (1, n), padding=(0, (n - 1) // 2)),
            ConvBN(channels_config[2], channels_config[2], (n, 1), padding=((n - 1) // 2, 0)),
            ConvBN(channels_config[2], channels_config[3], 3, 2, 0),
        ))
        self.add_module("p3", nn.MaxPool2d(3, 2, 0))
        
    
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x)
        ], 1)
        
        
    
class InceptionV3(nn.Module):
        
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()
        self.add_module("stem", nn.Sequential(
            ConvBN(3, 32, 3, 2),
            ConvBN(32, 32, 3, 1),
            ConvBN(32, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            ConvBN(64, 80, 1),
            ConvBN(80, 192, 3, 1),
            nn.MaxPool2d(3, 2)
        ))
        self.add_module("mix0", Inception_A(192, (64, 48, 64, 64, 96, 32)))
        self.add_module("mix1", Inception_A(256, (64, 48, 64, 64, 96, 64)))
        self.add_module("mix2", Inception_A(288, (64, 48, 64, 64, 96, 64)))
        self.add_module("mix3", Reduction_A(288, (384, 64, 96)))
        
        self.add_module("mix4", Inception_B(768, (192, 128, 192, 128, 192, 192)))
        self.add_module("mix5", Inception_B(768, (192, 160, 192, 160, 192, 192)))
        self.add_module("mix6", Inception_B(768, (192, 160, 192, 160, 192, 192)))
        self.add_module("mix7", Inception_B(768, (192, 192, 192, 192, 192, 192)))
        self.add_module("mix8", Reduction_B(768, (192, 320, 192, 192)))
        
        self.add_module("mix9", Inception_C(1280, (320, 384, 384, 448, 384, 192)))
        self.add_module("mix10", Inception_C(1280, (320, 384, 384, 448, 384, 192)))
        
        self.add_module("classifier", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, num_classes),
            nn.Softmax()
        ))
    
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    
if __name__ == "__main__":
    model = InceptionV3()
    summary(model, (3, 299, 299))
    
    
            
        
        
        
        