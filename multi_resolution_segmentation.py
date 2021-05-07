import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler
from efficientnet_pytorch.model import EfficientNet
from segmentation import to_device, train_or_eval, num_channels, cam_resolution
from utils import transform, train_transform

import argparse

class ClassActivationMap2d(nn.Module):
    '''Generates a class activation map from the input tensor (format: N * C * H * W).'''

    def __init__(self, in_channels: int, out_channels: int, input_res, output_res=None):
        super(ClassActivationMap2d, self).__init__()
        # Linear (1x1 convolution) layer to transform input feature maps into CAMs
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Average pooling if not generating CAM
        self.avgpool = nn.AvgPool2d(input_res)
        # Upsample to a certain size (optional)
        self.upsample = nn.Upsample(size=output_res) if output_res else None
    
    def forward(self, inputs, return_cam=False):
        if return_cam:
            # If generating CAM, apply the linear transformation then upsample
            out = self.linear(inputs)
            return self.upsample(out) if self.upsample else out
        else:
            # If not generating CAM, average pool the input to size 1x1, then apply the linear transformation
            avg = self.avgpool(inputs)
            return self.linear(avg)

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
        endpoints = kwargs.get('endpoints', [1, 2, 3, 4])
        pos_class_weight = kwargs.get('pos_class_weight', 2.0)
        # Use EfficientNet classifier as a backbone
        if from_pretrained:
            self.backbone = EfficientNet.from_pretrained(backbone, num_classes=2)
        else:
            self.backbone = EfficientNet.from_name(backbone, num_classes=2)
        
        # Create a segmentation branch and CAM layer for each endpoint
        self.seg_branches = nn.ModuleDict()
        self.activation_map = nn.ModuleDict()
        for endpoint in endpoints:
            # Create seg branch for endpoint
            in_channels = num_channels(endpoint=endpoint)
            seg_branch = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.seg_branches[f'reduction_{endpoint}'] = seg_branch
            # Create CAM layer for endpoint
            in_res = cam_resolution(endpoint=endpoint)
            self.activation_map[f'reduction_{endpoint}'] = ClassActivationMap2d(16, 2, in_res, 112)

        # Softmax over channels if outputting CAM
        self.softmax = nn.Softmax(dim=1)
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
                sub_cam = self.activation_map[endpoint_name](sub_cam, return_cam=return_cam)
                sub_cams.append(sub_cam)
        # Sum all CAMs together
        stacked_cams = torch.stack(sub_cams, dim=0)
        cam = torch.sum(stacked_cams, dim=0, keepdim=False)
        # Return either scores or activation map
        # Each Conv2D layer returns an output of dimension (N, C, H, W)
        if return_cam:
            # Compute softmax channel-wise and output only the positive class
            # probability for each pixel
            return self.softmax(cam)[:, 1]
        else:
            # Remove the (H, W) dimensions
            return torch.squeeze(cam)

    def freeze_backbone(self):
        '''Freezes the backbone.'''
        self.backbone.requires_grad_(False)

    def to_save_file(self, save_file):
        '''Saves this model to a save file in .pt format.
        Both the constructor parameters and the state_dict are saved.'''
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


def train_multi_segmentation(model: MultiResolutionSegmentation,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             optimizer: optim.Optimizer,
                             scaler: GradScaler = None,
                             num_epochs: int = 3):
    # Don't train the backbone
    model.freeze_backbone()
    # Train the segmentation part of the neural network
    for epoch in trange(num_epochs, desc=f'Training'):
        train_or_eval(model, train_loader, optimizer, scaler)
    # Evaluate on validation set once at the end
    return train_or_eval(model, val_loader, scaler=scaler)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and store the model')
    parser.add_argument('-o', '--out', metavar='model.pt', default='model.pt')
    parser.add_argument('-w', '--pos-class-weight', type=float, default=8.0)
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

    model = to_device(MultiResolutionSegmentation(pos_class_weight=args.pos_class_weight))
    # Use RMSProp parameters from the DeepSolar paper (alpha = second moment discount rate)
    # except for learning rate decay and epsilon
    optimizer = optim.RMSprop(model.parameters(), alpha=0.9, momentum=0.9, eps=0.001, lr=1e-3)
    # optimizer = optim.Adam(model.parameters()) # betas =(0.9, 0.9)
    
    scaler = GradScaler() if args.mixed_precision else None
    train_multi_segmentation(model, train_loader, val_loader, optimizer, scaler, num_epochs=args.num_epochs)

    model.to_save_file(args.out)

if __name__ == '__main__':
    main()
