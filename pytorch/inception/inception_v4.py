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
    
    
class InputStem(nn.Module):
    
    def __init__(self) -> None:
        super(InputStem, self).__init__()
        self.stem1 = nn.Sequential(
            ConvBN(3, 32, 3, 2),
            ConvBN(32, 32, 3, 1),
            ConvBN(32, 64, 3, 1, padding=1)
        )
        self.b1 = nn.MaxPool2d(3, 2, 0)
        self.b2 = ConvBN(64, 96, 3, 2, 0)
        
        self.s1 = nn.Sequential(
            ConvBN(160, 64, 1),
            ConvBN(64, 96, 3)
        )
        self.s2 = nn.Sequential(
            ConvBN(160, 64, 1),
            ConvBN(64, 64, (1, 7), padding=(0, 3)),
            ConvBN(64, 64, (7, 1), padding=(3, 0)),
            ConvBN(64, 96, 3)
        )
        self.p1 = ConvBN(192, 192, 3, 2)
        self.p2 = nn.MaxPool2d(3, 2, 0)
        
    def forward(self, x):
        x = self.stem1(x)
        x = torch.cat([self.b1(x), self.b2(x)], 1)
        x = torch.cat([self.s1(x), self.s2(x)], 1)
        x = torch.cat([self.p1(x), self.p2(x)], 1)
        return x
    
class Inception_A(nn.Module):
    
    def __init__(self, in_channels):
        # out_channels:  384
        super(Inception_A, self).__init__()
        self.add_module("p1", ConvBN(in_channels, 96, 1))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, 64, 1),
            ConvBN(64, 96, 3, padding=1)
        ))
        self.add_module("p3", nn.Sequential(
            ConvBN(in_channels, 64, 1),
            ConvBN(64, 96, 3, padding=1),
            ConvBN(96, 96, 3, padding=1)
        ))
        self.add_module("p4", nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBN(in_channels, 96, 1)
        ))
        
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x),
            self.p4(x)
        ], 1)
        

class Inception_B(nn.Module):
    
    def __init__(self, in_channels, n=7):
        # out_channels:  1024
        super(Inception_B, self).__init__()
        self.add_module("p1", ConvBN(in_channels, 384, 1))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, 192, 1),
            ConvBN(192, 224, (1, 3), padding=(0, 1)),
            ConvBN(224, 256, (3, 1), padding=(1, 0))
        ))
        self.add_module("p3", nn.Sequential(
            ConvBN(in_channels, 192, 1),
            ConvBN(192, 192, (n, 1), padding=((n - 1) // 2, 0)),
            ConvBN(192, 224, (1, n), padding=(0, (n - 1) // 2)),
            ConvBN(224, 224, (n, 1), padding=((n - 1) // 2, 0)),
            ConvBN(224, 256, (1, n), padding=(0, (n - 1) // 2)),
        )) 
        self.add_module("p4", nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBN(in_channels, 128, 1)
        ))
    
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x),
            self.p4(x)
        ], 1)
        

class Inception_C(nn.Module):
    
    def __init__(self, in_channels):
        # out_channels:  1536
        super(Inception_C, self).__init__()
        self.p1 = ConvBN(in_channels, 256, 1)
        self.p2 = ConvBN(in_channels, 384, 1)
        self.p21 = ConvBN(384, 256, (1, 3), padding=(0, 1))
        self.p22 = ConvBN(384, 256, (3, 1), padding=(1, 0))
        self.p3 =  nn.Sequential(
            ConvBN(in_channels, 384, 1),
            ConvBN(384, 448, (3, 1), padding=(1, 0)),
            ConvBN(448, 512, (1, 3), padding=(0, 1)),  
        )
        self.p31 = ConvBN(512, 256, (1, 3), padding=(0, 1))
        self.p32 = ConvBN(512, 256, (3, 1), padding=(1, 0))
        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBN(in_channels, 256, 1)
        )
        
    def forward(self, x):
        h2 = self.p2(x)
        h3 = self.p3(x)
        return torch.cat([
            self.p1(x),
            torch.cat([self.p21(h2), self.p22(h2)], 1),
            torch.cat([self.p31(h3), self.p32(h3)], 1),
            self.p4(x)
        ], 1)
        

class Reduction_A(nn.Module):

    def __init__(self, in_channels):
        # out_channels:  640 + in_channels(384) = 
        super(Reduction_A, self).__init__()
        self.add_module("p1", ConvBN(in_channels, 384, 3, 2, 0))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, 192, 3, 1, 1),
            ConvBN(192, 224, 3, 1, 1),
            ConvBN(224, 256, 3, 2, 0)
        ))
        self.add_module("p3", nn.MaxPool2d(3, 2, 0))
        
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x)
        ], 1)
        

class Reduction_B(nn.Module):
    
    def __init__(self, in_channels, n=7):
        # out_channels:  512 + in_channels
        super(Reduction_B, self).__init__()
        self.add_module("p1", nn.Sequential(
            ConvBN(in_channels, 192, 3, 1, 1),
            ConvBN(192, 192, 3, 2, 0),
        ))
        self.add_module("p2", nn.Sequential(
            ConvBN(in_channels, 256, 3, 1, 1),
            ConvBN(256, 256, (1, n), padding=(0, (n - 1) // 2)),
            ConvBN(256, 320, (n, 1), padding=((n - 1) // 2, 0)),
            ConvBN(320, 320, 3, 2, 0),
        ))
        self.add_module("p3", nn.MaxPool2d(3, 2, 0))
        
    
    def forward(self, x):
        return torch.cat([
            self.p1(x),
            self.p2(x),
            self.p3(x)
        ], 1)
        
        
    
class InceptionV4(nn.Module):
        
    def __init__(self, num_classes=1000):
        super(InceptionV4, self).__init__()
        self.add_module("stem", InputStem())
        
        for i in range(4):
            self.add_module("inception_a{}".format(i+1), Inception_A(384))
            
        self.add_module("reduction_a", Reduction_A(384))
        
        for i in range(7):
            self.add_module("inception_b{}".format(i+1), Inception_B(1024))
        
        self.add_module("reduction_b", Reduction_B(1024))
        
        for i in range(3):
            self.add_module("inception_c{}".format(i+1), Inception_C(1536))
        
        self.add_module("classifier", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.2),
            nn.Flatten(1),
            nn.Linear(1536, num_classes),
            nn.Softmax(dim=1)
        ))
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    
if __name__ == "__main__":
    model = InceptionV4()
    summary(model, (3, 299, 299))
    
    
            
        
        
        
        