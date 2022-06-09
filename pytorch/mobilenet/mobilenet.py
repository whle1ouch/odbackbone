import torch
import torch.nn as nn
from torchsummary import summary



class MobileBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileBlock, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, padding=1, 
                      bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.m(x)


class MobileBlockV2(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(MobileBlockV2, self).__init__()
        assert isinstance(expansion, int) == True 
        "expansion must be int"
        assert expansion > 0 
        "expansion must be positive"
        mid_channels = in_channels * expansion
        if expansion > 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels * 6),
                nn.ReLU(inplace=True))
        else:
            self.expand = nn.Sequential()

        self.m = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, padding=1, bias=False, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = stride == 1 and in_channels == out_channels
        
    def forward(self, x):
        out = self.m(self.expand(x))
        if self.shortcut:
            out = out + x
        return out
        
        
class MobileNet(nn.Module):
    
    def __init__(self, num_class=1000):
        super(MobileNet, self).__init__()
        self.add_module("input", nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ))
        self.add_module("block1", MobileBlock(32, 64, 1))
        self.add_module("block2", MobileBlock(64, 128, 2))
        self.add_module("block3", MobileBlock(128, 128, 1))
        self.add_module("block4", MobileBlock(128, 256, 2))
        self.add_module("block5", MobileBlock(256, 256, 1))
        self.add_module("block6", MobileBlock(256, 512, 2))
        self.add_module("block7", MobileBlock(512, 512, 1))
        self.add_module("block8", MobileBlock(512, 512, 1))
        self.add_module("block9", MobileBlock(512, 512, 1))
        self.add_module("block10", MobileBlock(512, 512, 1))
        self.add_module("block11", MobileBlock(512, 512, 1))
        self.add_module("block12", MobileBlock(512, 1024, 2))
        self.add_module("block13", MobileBlock(1024, 1024, 1))
        
        self.add_module("output", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.2),
            nn.Conv2d(1024, num_class, 1),
            nn.Flatten(1),
            nn.Softmax(dim=1)
        ))
        
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class MobileNetV2(nn.Module):
    
    def __init__(self, num_class=1000):
        super(MobileNetV2, self).__init__()
        self.add_module("input", nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ))
        self.add_module("block1", MobileBlockV2(32, 16, 1, expansion=1))
        self.add_module("block2", MobileBlockV2(16, 24, 2))
        self.add_module("block3", MobileBlockV2(24, 24, 1))
        self.add_module("block4", MobileBlockV2(24, 32, 2))
        self.add_module("block5", MobileBlockV2(32, 32, 1))
        self.add_module("block6", MobileBlockV2(32, 32, 1))
        self.add_module("block7", MobileBlockV2(32, 64, 2))
        self.add_module("block8", MobileBlockV2(64, 64, 1))
        self.add_module("block9", MobileBlockV2(64, 64, 1))
        self.add_module("block10", MobileBlockV2(64, 64, 1))
        self.add_module("block11", MobileBlockV2(64, 96, 1))
        self.add_module("block12", MobileBlockV2(96, 96, 1))
        self.add_module("block13", MobileBlockV2(96, 96, 1))
        self.add_module("block14", MobileBlockV2(96, 160, 2))
        self.add_module("block15", MobileBlockV2(160, 160, 1))
        self.add_module("block16", MobileBlockV2(160, 160, 1))
        self.add_module("block17", MobileBlockV2(160, 320, 1))
        
    
        self.add_module("output", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.2),
            nn.Conv2d(320, num_class, 1),
            nn.Flatten(1),
            nn.Softmax(dim=1)
        ))
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
        
if __name__ == "__main__":
    model = MobileNet()
    summary(model, (3, 224, 224))