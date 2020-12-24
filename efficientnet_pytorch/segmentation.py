# TODO Implement CAM-GLWT for EfficientNet
import torch
from torch import nn
from torch.nn import functional as F

from .model import EfficientNet

class EfficientNetSegmentation(nn.Module):
    def __init__(self, backbone='efficientnet-b0', endpoint=1):
        super().__init__()
        # Use EfficientNet classifier as a backbone
        self.backbone = EfficientNet.from_pretrained(backbone)
        # Use endpoint from reduction level i in [1, 2, 3, 4, 5]
        if endpoint in range(1, 6):
            self.endpoint = f'reduction_{endpoint}'
        else:
            raise ValueError('endpoint must be between 1 and 5, inclusive')
    
    def forward(self, batch):
        pass
