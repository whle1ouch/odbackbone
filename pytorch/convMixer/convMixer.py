
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class ActBn(nn.Module):
    
    def __init__(self, module, h):
        super().__init__()
        assert isinstance(module, nn.Module)
        self.m = module
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(h)
    
    def forward(self, x):
        x = self.m(x)
        x = self.gelu(x)
        x = self.bn(x)
        return x


class Residual(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        assert isinstance(module, nn.Module)
        self.m = module
    
    def forward(self, x):
        y = self.m(x)
        return x + y



class ConvMixer(nn.Module):
    
    def __init__(self, h, depth, kernel_size=9, patch_size=7, include_top=True, n_classses=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, h, patch_size, stride=patch_size)
        
        self.convs = nn.Sequential(
            *[nn.Sequential(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"), h)), 
                            ActBn(nn.Conv2d(h, h, 1), h)) 
              for _ in range(depth)]
        )
        if include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten(),
                nn.Linear(h, n_classses)
            )
        else:
            self.classifier = nn.Identity()
        
    def forward(self, x):
        x = self.stem(x)
        x = self.convs(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    model = ConvMixer(512, 1)
    summary(model, (3, 224, 224))
        
        