import torch 
import torch.nn as nn
from torchsummary import summary


class FPN(nn.Module):
    
    def __init__(self):
        super(FPN, self).__init__()
        self.fn1 = self._make_block(3, 32, 2)
        self.conv1 = nn.Conv2d(32, 256, 1)
        self.fn2 = self._make_block(32, 64, 2)
        self.conv2 = nn.Conv2d(64, 256, 1)
        self.fn3 = self._make_block(64, 128, 4)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.fn4 = self._make_block(128, 256, 4)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.fn5 = self._make_block(256, 512, 4)
        self.conv5 = nn.Conv2d(512, 256, 1)
        
    def forward(self, x):
        f1 = self.fn1(x)
        f2 = self.fn2(f1)
        f3 = self.fn3(f2)
        f4 = self.fn4(f3)
        f5 = self.fn5(f4)
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)
        f5 = self.conv5(f5)
        return f1, f2, f3, f4, f5
    
    
    def _make_block(self, in_channels, out_channels, num_conv, expand=1):
        filters_ = int(out_channels * expand)
        layers = []
        relu = nn.ReLU(inplace=True)
        layers.append(nn.Conv2d(in_channels, filters_, 3, 1, 1))
        layers.append(relu)
        for _ in range(num_conv-1):
            layers.append(nn.Conv2d(filters_, filters_, 3, 1, 1))
            layers.append(relu)
        layers.append(nn.Conv2d(filters_, out_channels, 3, 2, 1))
        layers.append(relu)
        return nn.Sequential(*layers)
    
    
if __name__ == "__main__":
    model = FPN()
    summary(model, (3, 208, 208))