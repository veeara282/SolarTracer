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
        # Create some number of Conv2d's followed by an AvgPool2d and Linear
        self.conv2ds = []
        # TODO Extract channel count from .utils module
        # Also, could use depthwise separable convolution instead (see MobileNet paper) --
        # it's what EfficientNet uses
        self.conv2ds.append(nn.Conv2d(16, 16, 3))
        self.avgpool = nn.AvgPool2d()
        self.linear = nn.Linear(16, 2)

    def forward(self, inputs, return_cam=False):
        # Extract hidden state from middle of EfficientNet
        hidden_state = self.backbone.extract_endpoints(inputs)[self.endpoint]
        # Apply Conv2d layers in succession
        for conv2d in self.conv2ds:
            hidden_state = conv2d(hidden_state)
        # Return either scores or activation map
        if return_cam:
            class_activation_map = self.linear(hidden_state)
            return class_activation_map
        else:
            class_scores = self.linear(self.avgpool(hidden_state))
            return class_scores
    
    def freeze_layers(self):
        '''Freezes the backbone and all current Conv2d layers.'''
        self.backbone.requires_grad_(False)
        for conv2d in self.conv2ds:
            conv2d.requires_grad_(False)

    def add_new_layer(self):
        '''Adds a new Conv2d layer and resets the Linear layer.'''
        self.conv2ds.append(nn.Conv2d(16, 16, 3))
        self.linear.reset_parameters()
