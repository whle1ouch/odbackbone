import math
import torch
import torch.nn as nn
from torchsummary import summary


class DarknetConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, bias=False):
        super(DarknetConv2d, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x
    
    
class Focus(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
    
    def forward(self, x):
        x = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))
        return x
    

class SPPF(nn.Module):
    
    def __init__(self, in_channels, out_channels, pool_size=5):
        super(SPPF, self).__init__()
        mid_channels = in_channels // 2
        self.conv1 = DarknetConv2d(in_channels, mid_channels, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, padding=pool_size //2)
        self.conv2 = DarknetConv2d(mid_channels * 4, out_channels, 1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.conv2(torch.cat([y1, y2, x, self.pool(y2)], dim=1))
    
class CSPDarkRes(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_blocks, shortcut=True, expansion=0.5):
        super(CSPDarkRes, self).__init__()
        mid_channels = int(out_channels * expansion)
        self.shortcut = shortcut
        self.concat = DarknetConv2d(in_channels, mid_channels, 1)
        self.blocks_before = DarknetConv2d(in_channels, mid_channels, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                DarknetConv2d(mid_channels, mid_channels, 1),
                DarknetConv2d(mid_channels, mid_channels, 3),
            )
            for _ in range(num_blocks)
        ])
        
        self.blocks_final = DarknetConv2d(2 * mid_channels, out_channels, 1)
    
    def forward(self, x):
        y1 = self.concat(x)
        x = self.blocks_before(x)
        for block in self.blocks:
            if self.shortcut:
                x = block(x) + x
            else:
                x = block(x)
        x = torch.cat([x, y1], dim=1)
        return self.blocks_final(x)
    
def depth_gain(n, depth_multiple):
    return max(round(n * depth_multiple), 1) if n > 1 else n

def width_gain(n, divisor=8):
    return int(math.ceil(n / divisor) * divisor)
        

class CSPDarknet53(nn.Module):
    
    width_config = (64, 128, 256, 512, 1024)
    depth_config = (3, 6, 9, 3)
    
    def __init__(self, depth_multiple, width_multiple):
        super(CSPDarknet53, self).__init__()
        width_config = [width_gain(n * width_multiple) for n in self.width_config]
        depth_config = [depth_gain(n, depth_multiple) for n in self.depth_config]
        
        self.stem1 = nn.Sequential(
            DarknetConv2d(3, width_config[0], 6, 2, 2),
            DarknetConv2d(width_config[0], width_config[1], 3, 2, 1),
            CSPDarkRes( width_config[1],  width_config[1], depth_config[0]),
            DarknetConv2d(width_config[1], width_config[2], 3, 2, 1),
            CSPDarkRes(width_config[2], width_config[2], depth_config[1]),
        )
        
        self.stem2 = nn.Sequential(
            DarknetConv2d(width_config[2], width_config[3], 3, 2, 1),
            CSPDarkRes(width_config[3], width_config[3], depth_config[2])
        )
        self.stem3 = nn.Sequential(
            DarknetConv2d(width_config[3], width_config[4], 3, 2, 1),
            CSPDarkRes(width_config[4], width_config[4], depth_config[3]),
            SPPF(width_config[4], width_config[4], 5)
        )

        
    def forward(self, x):
        f1 = self.stem1(x)
        f2 = self.stem2(f1)
        f3 = self.stem3(f2)
        return [f1, f2, f3]
    
def CspDarkNetSmall():
    return CSPDarknet53(0.33, 0.50)

def CspDarkNetXtreme():
    return CSPDarknet53(1.33, 1.25)

def CspDarkNetLarge():
    return CSPDarknet53(1.0, 1.0)

def CspDarkNetMedium(inuts):
    return CSPDarknet53(0.67, 0.75)

def CspDarkNetNano():
    return CSPDarknet53(0.33, 0.25)
    
    
if __name__ == "__main__":
    model = CspDarkNetSmall()
    summary(model, input_size=(3, 608, 608))