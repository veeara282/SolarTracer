import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder

from .segmentation import EfficientNetSegmentation, to_device, transform, train_or_eval
import argparse
import os
import re

def classification_test_set(root, **kwargs):
    # Enumerate the direct children of SPI_eval
    subdirs = [entry.path for entry in os.scandir(root) if entry.is_dir()]
    
    # Create an ImageFolder for each one and exclude the *_true_seg.png files
    def is_input_image(filename: str) -> bool:
        return bool(re.match('\\d+\\.\\w+', filename))
    subsets = [ImageFolder(subdir, is_valid_file=is_input_image, **kwargs) for subdir in subdirs]

    # Add them all to a ConcatDataset
    return ConcatDataset(subsets)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the model on the test set')
    parser.add_argument('model', metavar='model.pth', default='model.pth')
    parser.add_argument('-b', '--batch-size', type=int, default=48)
    return parser.parse_args()

def main():
    args = parse_args()

    # Read model file
    model = to_device(EfficientNetSegmentation())
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Set up test set
    test_set = ImageFolder(root='./SPI_eval/', transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Evaluate
    train_or_eval(model, test_loader)

if __name__ == '__main__':
    main()
