import torch
import torch.nn as nn
from torch.autograd import Variable

# https://jennaweng0621.pixnet.net/blog/post/403589795-%5Bpytorch%5D-vgg%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%E7%B5%90%E6%A7%8B%28vgg11%2C-vgg13%2C-vgg16%2C-vgg19%29 
class VGG(nn.Module):
    
    def __init__(self, in_channels, cfg):
        
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg)
        self.fc = nn.Linear(5 * 5 * 128, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=0),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    
    in_channels = 1
    cfgs = {
        'VGG_gray': [64, 'M', 128, 'M'],
    }
    
    model = VGG(in_channels, cfgs["VGG_gray"])
    model(torch.randn((2, 1, 28, 28)))