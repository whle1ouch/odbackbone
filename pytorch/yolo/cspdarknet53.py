import torch
import torch.nn as nn
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
        self.mish = nn.Mish(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x
    

class CSPDarkRes(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_blocks, expansion=0.5):
        super(CSPDarkRes, self).__init__()
        mid_channels = int(out_channels * expansion)
        self.stem = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            DarknetConv2d(in_channels, out_channels, 3, stride=2),
        )
        self.shorcut = DarknetConv2d(out_channels, mid_channels, 1)
        self.blocks_before = DarknetConv2d(out_channels, mid_channels, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                DarknetConv2d(mid_channels, out_channels // 2, 1),
                DarknetConv2d(out_channels // 2, mid_channels, 3),
            )
            for _ in range(num_blocks)
        ])
        self.blocks_after = DarknetConv2d(mid_channels, mid_channels, 1)
        self.blocks_final = DarknetConv2d(2 * mid_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.stem(x)
        shortcut = self.shorcut(x)
        x = self.blocks_before(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.blocks_after(x)
        x = self.blocks_final(torch.cat([x, shortcut], 1))
        return x
        

class CSPDarknet53(nn.Module):
    
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.conv1 = DarknetConv2d(3, 32, 3, stride=1)
        self.res2 = CSPDarkRes(32, 64, 1, expansion=1)
        self.res3 = CSPDarkRes(64, 128, 2)
        self.res4 = CSPDarkRes(128, 256, 8)
        self.res5 = CSPDarkRes(256, 512, 8)
        self.res6 = CSPDarkRes(512, 1024, 4)
        
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
    model = CSPDarknet53()
    summary(model, input_size=(3, 416, 416))