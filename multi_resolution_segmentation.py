import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch.model import EfficientNet

import argparse

class MultiResolutionSegmentation(nn.Module):
    '''Generates a class activation map by combining multiple CAMs at different resolutions.'''
    # TODO Implement
    # Useful classes:
    # - torch.nn.Upsample
    # - torch.nn.ModuleList/ModuleDict
    # - ParameterDict?

    def __init__(self, from_pretrained=True, **kwargs):
        super(MultiResolutionSegmentation, self).__init__()
        # # Extract arguments and save the params
        self.constructor_params = kwargs
        backbone = kwargs.get('backbone', 'efficientnet-b0')
        # endpoints = kwargs.get('endpoint', [1, 2, 3, 4])
        pos_class_weight = kwargs.get('pos_class_weight', 2.0)
        # Use EfficientNet classifier as a backbone
        if from_pretrained:
            self.backbone = EfficientNet.from_pretrained(backbone, num_classes=2)
        else:
            self.backbone = EfficientNet.from_name(backbone, num_classes=2)
        
        # Create a segmentation branch for each endpoint
        self.seg_branches = nn.ModuleDict()
        # Reduction level 3
        seg_branch_3 = nn.Sequential(
            nn.Conv2d(40, 16, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1)
        )
        self.seg_branches['reduction_3'] = seg_branch_3
        # Upsample all CAMs to 112x112
        self.upsampler = nn.Upsample(size=112, mode='nearest')
        # Average pooling if not outputting CAM
        self.avgpool = nn.AvgPool2d(112)
        # Softmax if outputting CAM
        self.softmax = nn.Softmax(dim=-1)
        # Loss function
        class_weights = torch.FloatTensor([1.0, pos_class_weight])
        self.loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, return_cam=False):
        # Extract hidden states from EfficientNet endpoints
        hidden_states = self.backbone.extract_endpoints(inputs)
        # Sub-class activation maps
        sub_cams = []
        # Put each one through its respective segmentation branch if it exists,
        # then upsample to 112x112
        for endpoint_name in hidden_states.keys():
            # Get seg branch
            if endpoint_name in self.seg_branches:
                seg_branch = self.seg_branches[endpoint_name]
                hidden_state = hidden_states[endpoint_name]
                sub_cam = seg_branch(hidden_state)
                sub_cam = self.upsampler(sub_cam)
                sub_cams.append(sub_cam)
        # Sum all CAMs together
        stacked_cams = torch.stack(sub_cams, dim=0)
        cam = torch.sum(stacked_cams, dim=0, keepdim=False)
        # Return either scores or activation map
        if return_cam:
            return self.softmax(cam)[..., 1]
        else:
            avgpool = torch.squeeze(self.avgpool(cam))
            return avgpool

