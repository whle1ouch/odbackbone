import torch
import torch.nn as nn
from torchsummary import summary

class ResBottle(nn.Module):
    
    def __init__(self, inplane, outplane, stride=1):
        super(ResBottle, self).__init__()
        midplane = outplane // 4
        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)
        
        self.conv3 = nn.Conv2d(midplane, outplane, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplane)
        
        if inplane == outplane:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplane)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = out + self.shortcut(x)

        return out 
    

class ResBottleV2(nn.Module):
    
    def __init__(self, inplane, outplane, stride=1):
        super(ResBottleV2, self).__init__()
        midplane = outplane // 4
        self.bn0 = nn.BatchNorm2d(inplane)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)
        
        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)
        
        self.conv3 = nn.Conv2d(midplane, outplane, kernel_size=1, bias=False)
        
        if inplane == outplane:
            if stride > 1:
                self.shortcut = nn.MaxPool2d(1, stride=stride)
            else:
                self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv2d(inplane, outplane, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        pre = self.bn0(x)
        pre = self.relu(pre)
        
        out = self.conv1(pre)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = out + self.shortcut(pre)
        return out
    
class ResNet(nn.Module):
    
    def __init__(self, residue_nums, strides, preact=False, num_classes=1000):
        super(ResNet, self).__init__()
        assert len(residue_nums) == len(strides) 
        "residue_nums and strides must be the same length"
        inplane = 64
        self.add_module('conv', nn.Conv2d(3, inplane, kernel_size=7, stride=2, padding=3, bias=False))
        if not preact:
            self.add_module('bn', nn.BatchNorm2d(inplane))
            self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        for i, (residue_num, stride) in enumerate(zip(residue_nums, strides)):
            outplane = inplane * 2
            self.add_module(f'resblock{i+1}', self._make_layers(inplane, outplane, residue_num, stride, preact))
            inplane = outplane
        
        if preact:
           self.add_module('bn', nn.BatchNorm2d(outplane))
           self.add_module("relu", nn.ReLU(inplace=True))
           
        self.add_module('classfier', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(outplane, num_classes),
            nn.Softmax(dim=1)
        ))

        

    def _make_layers(self, inplane, outplane, residue_num, stride, preact):
        layers = []
        if preact:
            layers.append(ResBottleV2(inplane, outplane))
            (layers.append(ResBottleV2(outplane, outplane)) for _ in range(residue_num - 2))
            layers.append(ResBottleV2(outplane, outplane, stride=stride))
        else:
            layers.append(ResBottle(inplane, outplane, stride=stride))
            (layers.append(ResBottle(outplane, outplane, stride=stride)) for _ in range(residue_num - 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        for name, module in self._modules.items():
            x = module(x)
        return x
    
    
def resnet50():
    return ResNet([3, 4, 6, 3], [1, 2, 2, 2])

def resnet50v2():
    return ResNet([3, 4, 6, 3], [2, 2, 2, 1], preact=True)

def resnet101():
    return ResNet([3, 4, 23, 3], [1, 2, 2, 2])

def resnet101v2():
    return ResNet([3, 4, 23, 3], [2, 2, 2, 1], preact=True)

def resnet152():
    return ResNet([3, 8, 36, 3], [1, 2, 2, 2])

def resnet152v2():
    return ResNet([3, 8, 36, 3], [2, 2, 2, 1], preact=True)


if __name__ == "__main__":
    model = resnet50()
    summary(model, (3, 224, 224))