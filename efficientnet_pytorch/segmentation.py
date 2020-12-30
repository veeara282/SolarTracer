# TODO Implement CAM-GLWT for EfficientNet
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from torchvision.datasets import ImageFolder
from torchvision import transforms

from .model import EfficientNet

class EfficientNetSegmentation(nn.Module):
    def __init__(self, backbone='efficientnet-b0', endpoint=1):
        super().__init__()
        # Use EfficientNet classifier as a backbone
        self.backbone = EfficientNet.from_pretrained(backbone, num_classes=2)
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
        self.avgpool = nn.AvgPool2d(112)
        self.linear = nn.Linear(16, 2)
        # Loss function

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
    
    def freeze_backbone(self):
        '''Freezes the backbone.'''
        self.backbone.requires_grad_(False)

    def freeze_conv2d_layers(self):
        '''Freezes all current Conv2d layers.'''
        for conv2d in self.conv2ds:
            conv2d.requires_grad_(False)

    def add_new_layer(self):
        '''Adds a new Conv2d layer and resets the Linear layer.'''
        self.conv2ds.append(nn.Conv2d(16, 16, 3))
        self.linear.reset_parameters()

def to_device(obj):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return obj.to(device)

def train_or_eval(model: nn.Module,
                  data_loader: DataLoader,
                  loss_criterion: nn.Module,
                  optimizer: optim.Optimizer,
                  train: bool = False):
    if train:
        model.train()
    else:
        model.eval()
    total = 0
    total_loss = 0
    correct = 0
    for batch in tqdm(data_loader, leave=False, desc=("Training Batches" if train else "Validation Batches")):
        inputs, labels = batch[0], batch[1]
        output = model(inputs)
        print(output.size())
        total += output.size()[0]
        predicted = torch.argmax(output, 1).cpu()
        correct += (labels == predicted).numpy().sum()
        if train:
            optimizer.zero_grad() 
            loss = loss_criterion(output, to_device(labels))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        else:
            total_loss += loss_criterion(output, to_device(labels)).item()
    total_loss /= total
    # Print accuracy
    print('Training Results:' if train else 'Validation Results:')
    print(f'Loss: {total_loss:.4f}')
    print(f'Correct: {correct} ({correct / total:.2%})')
    print(f'Total: {total}\n')


def train_segmentation(model: EfficientNetSegmentation,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       loss_criterion: nn.Module,
                       optimizer: optim.Optimizer):
    # Train the backbone
    num_epochs = 5
    for epoch in trange(num_epochs, desc='Train backbone'):
        train_or_eval(model.backbone, train_loader, loss_criterion, optimizer, train=True)
        train_or_eval(model.backbone, val_loader, loss_criterion, optimizer)
    # Train the segmentation branch
    model.freeze_backbone()
    for epoch in trange(num_epochs, desc='Train segmentation branch'):
        train_or_eval(model, train_loader, loss_criterion, optimizer, train=True)
        train_or_eval(model, train_loader, loss_criterion, optimizer)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = ImageFolder(root='./SPI_val/', transform=transform)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    
    val_set = ImageFolder(root='./SPI_eval/1/', transform=transform)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)

    model = EfficientNetSegmentation()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())
    
    train_segmentation(model, train_loader, val_loader, loss_criterion, optimizer)

if __name__ == '__main__':
    main()
