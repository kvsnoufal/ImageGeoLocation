
import torch
from torch import nn
from torch.utils import data
import torchvision
from config import Config

class GeoModel(nn.Module):
    def __init__(self):
        super(GeoModel, self).__init__()
        self.backbone = torchvision.models.resnext50_32x4d(pretrained=Config.PRETRAINED)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048,Config.NUM_CLASSES,bias=True),
        )
        if FEATURE_EXTRACTING:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
    def forward(self,x):
        x = self.backbone(x)
        return x