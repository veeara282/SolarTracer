import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch.model import EfficientNet

from .separable_conv2d import SeparableConv2d
import argparse

class MultiResolutionSegmentation(nn.Module):
    '''Generates a class activation map by combining multiple CAMs at different resolutions.'''
    # TODO Implement
    # Useful classes:
    # - torch.nn.Upsample
    # - SeparableConv2d (in this repo)
    # - torch.nn.ModuleList/ModuleDict
    # - ParameterDict?

    def __init__(self, from_pretrained=True, **kwargs):
        super(MultiResolutionSegmentation, self).__init__()

