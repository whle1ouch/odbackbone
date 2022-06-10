import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DarknetConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super(DarknetConv2d, self).__init__()
        if stride == 2:
            padding = 0
        else:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(0.1, inplace=True)
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.LeakyReLU(x)
        return x
    

class DarkRes(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_blocks):
        super(DarkRes, self).__init__()
        self.stem = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            DarknetConv2d(in_channels, out_channels, 3, stride=2),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                DarknetConv2d(out_channels, out_channels //2, 1),
                DarknetConv2d(out_channels //2, out_channels, 3),
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x) + x
        return x
        

class Darknet53(nn.Module):
    
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = DarknetConv2d(3, 32, 3, stride=1)
        self.res2 = DarkRes(32, 64, 1)
        self.res3 = DarkRes(64, 128, 2)
        self.res4 = DarkRes(128, 256, 8)
        self.res5 = DarkRes(256, 512, 8)
        self.res6 = DarkRes(512, 1024, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        f1 = x
        x = self.res5(x)
        f2 = x
        x = self.res6(x)
        f3 = x
        return [f1, f2, f3]
    
    
    
if __name__ == "__main__":
    model = Darknet53()
    summary(model, input_size=(3, 416, 416))