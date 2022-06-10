import torch
import torch.nn as nn
from torchsummary import summary


class SeparableConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.relu = nn.ReLU()
        self.stem0 = nn.Sequential(
            SeparableConv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
            SeparableConv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2, padding=1)
        )
        self.shortcut1 = self._make_shortcut(64, 128)
        self.stem1 = self._make_layers(64, 128)
        self.shortcut2 = self._make_shortcut(128, 256)
        self.stem2 = self._make_layers(128, 256)
        self.shortcut3 = self._make_shortcut(256, 728)
        self.stem3 = self._make_layers(256, 728)
        
    def _make_layers(self, in_channels, out_channels):
        return nn.Sequential(
            SeparableConv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu,
            SeparableConv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(3, 2, padding=1)
        )
        
    def _make_shortcut(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.stem0(x)
        x = self.stem1(x) + self.shortcut1(x)
        x = self.stem2(x) + self.shortcut2(x)
        x = self.stem3(x) + self.shortcut3(x)
        return x
    
class MiddleFlow(nn.Module):
    def __init__(self, num_block=8):
        super(MiddleFlow, self).__init__()
        for i in range(num_block):
            self.add_module(f"block{i}", self._make_layers(728, 728))
    

    def _make_layers(self, in_channels, out_channels):
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(
            relu,
            SeparableConv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            relu,
            SeparableConv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            relu,
            SeparableConv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        for module in self.children():
            x = x + module(x)
        return x
    
class ExitFlow(nn.Module):
    
    def __init__(self, num_classes=1000):
        super(ExitFlow, self).__init__()
        relu = nn.ReLU(inplace=True)
        self.step = nn.Sequential(
            relu,
            SeparableConv2d(728, 728, 3, 1, 1, bias=False),
            nn.BatchNorm2d(728),
            relu,
            SeparableConv2d(728, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, 2, 1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, 2, bias=False),
            nn.BatchNorm2d(1024)
        )
        
        self.classifier = nn.Sequential(
            SeparableConv2d(1024, 1536, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1536),
            relu,
            SeparableConv2d(1536, 2048, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2048),
            relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
            nn.Softmax()
        )
        
    def forward(self, x):
        x = self.step(x) + self.shortcut(x)
        x = self.classifier(x)
        return x
    
    
def Xeception(num_classes=1000):
    return nn.Sequential(
        EntryFlow(),
        MiddleFlow(),
        ExitFlow(num_classes))
    
if __name__ == "__main__":
    model = Xeception()
    summary(model, (3, 299, 299))