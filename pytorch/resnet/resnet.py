from turtle import forward
import torch
import torch.nn as nn

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
        self.bn2 = nn.BatchNorm2d(outplane)
        
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
        