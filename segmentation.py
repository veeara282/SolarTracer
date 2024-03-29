# TODO Implement CAM-GLWT for EfficientNet
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler

from efficientnet_pytorch.model import EfficientNet
import argparse

from utils import train_transform, transform, train_or_eval, to_device

def cam_resolution(backbone='efficientnet-b0', endpoint=1):
    if endpoint in range(1, 6):
        return 224 // (2 ** endpoint)
    else:
        raise ValueError('endpoint must be an integer between 1 and 5, inclusive')

def num_channels(backbone='efficientnet-b0', endpoint=1):
    if endpoint in range(1, 6):
        return [16, 24, 40, 112, 1280][endpoint - 1]
    else:
        raise ValueError('endpoint must be an integer between 1 and 5, inclusive')

class EfficientNetSegmentation(nn.Module):
    def __init__(self, from_pretrained=True, **kwargs):
        super().__init__()
        # Extract arguments and save the params
        self.constructor_params = kwargs
        backbone = kwargs.get('backbone', 'efficientnet-b0')
        endpoint = kwargs.get('endpoint', 1)
        pos_class_weight = kwargs.get('pos_class_weight', 2.0)
        num_layers = kwargs.get('num_layers', 0)
        # Use EfficientNet classifier as a backbone
        if from_pretrained:
            self.backbone = EfficientNet.from_pretrained(backbone, num_classes=2)
        else:
            self.backbone = EfficientNet.from_name(backbone, num_classes=2)
        # Use endpoint from reduction level i in [1, 2, 3, 4, 5]
        if endpoint in range(1, 6):
            self.endpoint = f'reduction_{endpoint}'
        else:
            raise ValueError('endpoint must be an integer between 1 and 5, inclusive')
        # Create some number of Conv2d's followed by an AvgPool2d and Linear
        self.conv2ds = nn.ModuleList()
        # TODO Extract channel count from .utils module
        # Also, could use depthwise separable convolution instead (see MobileNet paper) --
        # it's what EfficientNet uses
        self.avgpool = nn.AvgPool2d(cam_resolution(endpoint=endpoint))
        self.num_channels = num_channels(endpoint=endpoint)
        self.linear = nn.Linear(self.num_channels, 2)
        # If num_layers is provided, create the desired number of layers (this is needed to
        # load the state_dict properly)
        for _ in range(num_layers):
            self.add_conv2d_layer(reset_params=False)
        # Softmax for outputting the CAM
        self.softmax = nn.Softmax(dim=-1)
        # Loss function
        class_weights = torch.FloatTensor([1.0, pos_class_weight])
        self.loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, return_cam=False):
        # Extract hidden state from middle of EfficientNet
        hidden_state = self.backbone.extract_endpoints(inputs)[self.endpoint]
        # Apply Conv2d layers in succession
        for conv2d in self.conv2ds:
            hidden_state = conv2d(hidden_state)
        # Return either scores or activation map
        if return_cam:
            # Channel dimension needs to be at the end for nn.Linear to work
            hidden_state = torch.movedim(hidden_state, 1, -1)
            cam = self.linear(hidden_state)
            cam = self.softmax(cam)[..., 1]
            return cam
        else:
            avgpool = torch.squeeze(self.avgpool(hidden_state))
            class_scores = self.linear(avgpool)
            return class_scores
    
    def new_conv2d_layer(self):
        return to_device(nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1))

    def freeze_backbone(self):
        '''Freezes the backbone.'''
        self.backbone.requires_grad_(False)

    def freeze_conv2d_layers(self):
        '''Freezes all current Conv2d layers.'''
        for conv2d in self.conv2ds:
            conv2d.requires_grad_(False)

    def add_conv2d_layer(self, reset_params=True):
        '''Adds a new Conv2d layer and resets the Linear layer.'''
        self.conv2ds.append(self.new_conv2d_layer())
        if reset_params:
            self.linear.reset_parameters()
    
    def to_save_file(self, save_file):
        '''Saves this model to a save file in .pt format.
        Both the constructor parameters and the state_dict are saved.'''
        # Update num_layers as it is not updated when the number of layers changes
        self.constructor_params['num_layers'] = len(self.conv2ds)
        save_json = {
            'constructor_params': self.constructor_params,
            'state_dict': self.state_dict()
        }
        torch.save(save_json, save_file)
    
    @classmethod
    def from_save_file(cls, save_file):
        '''Creates a new model from an existing save file.
        The save file contains both the state_dict and constructor parameters needed to
        initialize the model correctly.'''
        save_json = torch.load(save_file)
        # Extract constructor params and state_dict
        params = save_json['constructor_params']
        state_dict = save_json['state_dict']
        # Create model according to params and load state_dict
        model = cls(from_pretrained=False, **params)
        model.load_state_dict(state_dict)
        return to_device(model)


def train_segmentation(model: EfficientNetSegmentation,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       optimizer: optim.Optimizer,
                       scaler: GradScaler = None):
    # Don't train the backbone
    model.freeze_backbone()
    # Number of layers to add and number of epochs per layer
    num_layers_branch = parse_args().num_layers_branch
    num_epochs = parse_args().num_epochs
    # Train the segmentation branch. At first, this branch has no conv2d layers and a linear layer.
    for layer_num in trange(num_layers_branch, desc='Build segmentation branch'):
        model.add_conv2d_layer()
        for epoch in trange(num_epochs, desc=f'Train branch layer {layer_num}'):
            train_or_eval(model, train_loader, optimizer, scaler)
        model.freeze_conv2d_layers()
    # Evaluate on validation set once at the end
    train_or_eval(model, val_loader, scaler=scaler)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and store the model')
    parser.add_argument('-o', '--out', metavar='model.pt', default='model.pt')
    parser.add_argument('-w', '--pos-class-weight', type=float, default=8.0)
    parser.add_argument('-l', '--num-layers-branch', type=int, default=3)
    parser.add_argument('-e', '--num-epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=48)
    parser.add_argument('-m', '--mixed-precision', action='store_true')
    parser.add_argument('--train-dir', default='./SPI_train/')
    parser.add_argument('--val-dir', default='./SPI_val/')
    return parser.parse_args()

def main():
    args = parse_args()

    train_set = ImageFolder(root=args.train_dir, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_set = ImageFolder(root=args.val_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = to_device(EfficientNetSegmentation(pos_class_weight=args.pos_class_weight))
    # Use RMSProp parameters from the DeepSolar paper (alpha = second moment discount rate)
    # except for learning rate decay and epsilon
    optimizer = optim.RMSprop(model.parameters(), alpha=0.9, momentum=0.9, eps=0.001, lr=1e-3)
    # optimizer = optim.Adam(model.parameters()) # betas =(0.9, 0.9)
    
    scaler = GradScaler() if args.mixed_precision else None
    train_segmentation(model, train_loader, val_loader, optimizer, scaler)

    model.to_save_file(args.out)

if __name__ == '__main__':
    main()
