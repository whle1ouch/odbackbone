import torch
import torch.nn as nn
from torchsummary import summary


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num):
        super(VGGBlock, self).__init__()
        self.add_module("conv0", self.add_layer(in_channels, out_channels))
        for i in range(1, conv_num):
            self.add_module(f"conv{i}", self.add_layer(out_channels, out_channels))
        self.add_module("pool", nn.MaxPool2d(2, 2))
        
        
    def add_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, x):
        for name, module in self._modules.items():
            x = module(x)
        return x

    
class VGG(nn.Module):
    
    
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
            nn.Softmax())
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
def vgg16():
    features = nn.Sequential(
        VGGBlock(3, 64, 2),
        VGGBlock(64, 128, 2),
        VGGBlock(128, 256, 3),
        VGGBlock(256, 512, 3),
        VGGBlock(512, 512, 3),
    )
    return VGG(features)
    
def vgg19():
    features = nn.Sequential(
        VGGBlock(3, 64, 2),
        VGGBlock(64, 128, 2),
        VGGBlock(128, 256, 4),
        VGGBlock(256, 512, 4),
        VGGBlock(512, 512, 4),
    )
    return VGG(features)

if __name__ == "__main__":
    model = vgg16()
    summary(model, (3, 224, 224))