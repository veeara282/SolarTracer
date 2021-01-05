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
    def __init__(self, backbone='efficientnet-b0', endpoint=1, pos_class_weight=2.0):
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
        self.avgpool = nn.AvgPool2d(112)
        self.linear = nn.Linear(16, 2)
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
            class_activation_map = self.linear(hidden_state)
            return class_activation_map
        else:
            avgpool = torch.squeeze(self.avgpool(hidden_state))
            class_scores = self.linear(avgpool)
            return class_scores
    
    def new_conv2d_layer(self):
        return nn.Conv2d(16, 16, 3, padding=1)

    def freeze_backbone(self):
        '''Freezes the backbone.'''
        self.backbone.requires_grad_(False)

    def freeze_conv2d_layers(self):
        '''Freezes all current Conv2d layers.'''
        for conv2d in self.conv2ds:
            conv2d.requires_grad_(False)

    def add_conv2d_layer(self):
        '''Adds a new Conv2d layer and resets the Linear layer.'''
        self.conv2ds.append(self.new_conv2d_layer())
        self.linear.reset_parameters()

def to_device(obj):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return obj.to(device)

def train_or_eval(model: nn.Module,
                  data_loader: DataLoader,
                  optimizer: optim.Optimizer,
                  train: bool = False):
    if train:
        model.train()
    else:
        model.eval()
    total = 0
    total_loss = 0
    correct = 0
    true_pos, all_true, all_pos = 0, 0, 0
    for batch in tqdm(data_loader, desc=("Training Batches" if train else "Validation Batches")):
        inputs, labels = batch[0], batch[1]
        output = model(to_device(inputs))
        total += output.size()[0]
        predicted = torch.argmax(output, 1).cpu()
        correct += (labels == predicted).numpy().sum()
        true_pos += (labels & predicted).numpy().sum()
        all_true += labels.numpy().sum()
        all_pos += predicted.numpy().sum()
        if train:
            optimizer.zero_grad() 
            loss = model.loss_criterion(output, to_device(labels))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        else:
            total_loss += model.loss_criterion(output, to_device(labels)).item()
    total_loss /= total
    precision = true_pos / all_pos
    recall = true_pos / all_true
    f1 = 2 * precision * recall / (precision + recall)
    # Print accuracy
    print('Training Results:' if train else 'Validation Results:')
    print(f'Loss: {total_loss:.4f}')
    print(f'Correct: {correct} ({correct / total:.2%})')
    print(f'Precision: {true_pos} / {all_pos} ({precision:.2%})')
    print(f'Recall: {true_pos} / {all_true} ({recall:.2%})')
    print(f'F1: {f1:.2%}')
    print(f'Total: {total}\n')


def train_segmentation(model: EfficientNetSegmentation,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       optimizer: optim.Optimizer):
    # Don't train the backbone
    model.freeze_backbone()
    # Number of layers to add and number of epochs per layer
    num_layers_branch = 3
    num_epochs = 3
    # Train the segmentation branch. At first, this branch has no conv2d layers and a linear layer.
    for layer_num in trange(num_layers_branch, desc='Build segmentation branch'):
        model.add_conv2d_layer()
        for epoch in trange(num_epochs, desc=f'Train branch layer {layer_num}'):
            train_or_eval(model, train_loader, optimizer, train=True)
            # train_or_eval(model, val_loader, optimizer)
        model.freeze_conv2d_layers()

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = ImageFolder(root='./SPI_val/', transform=transform)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    
    val_set = ImageFolder(root='./SPI_eval/1/', transform=transform)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)

    model = to_device(EfficientNetSegmentation(pos_class_weight=8.0))
    optimizer = optim.RMSprop(model.parameters())
    
    train_segmentation(model, train_loader, val_loader, optimizer)

if __name__ == '__main__':
    main()
