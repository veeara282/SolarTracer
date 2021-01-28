import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from .segmentation import EfficientNetSegmentation, to_device, transform, train_or_eval
import argparse
import os
import re

def classification_test_set(root, **kwargs):
    # Enumerate the direct children of SPI_eval
    subdirs = [entry.path for entry in os.scandir(root) if entry.is_dir()]
    
    # Create an ImageFolder for each one and exclude the *_true_seg.png files
    def is_input_image(filename: str) -> bool:
        # filename is the full path, not just the base name of the file
        # re.search matches anywhere in the string, whereas re.match only matches the beginning
        return bool(re.search(r'\d+\.\w+', filename))
    subsets = [ImageFolder(subdir, is_valid_file=is_input_image, **kwargs) for subdir in subdirs]

    # Add them all to a ConcatDataset
    return ConcatDataset(subsets)

def threshold(image_mask: torch.FloatTensor) -> torch.LongTensor:
    '''Converts a FloatTensor to a binary LongTensor using a threshold.'''
    return (image_mask > 0.6).to(torch.long)

# The transformation applied to image segmentation masks.
# Image masks are 8-bit grayscale PNG images (mode L), so transforms.Grayscale is a no-op.
target_transform = transforms.Compose([
    # no-op, but just in case the image format is different from expected
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Lambda(threshold)
])

class SegmentationTestSet(Dataset):
    def __init__(self, root, transform = transform, target_transform = target_transform):
        self.root = root
        # TODO create an index of (image, mask) file path pairs
        self.samples = []

    def __getitem__(self, index: int):
        pass

    def __len__(self):
        pass


def segmentation_test_set(root, **kwargs):
    # Enumerate the subfolders of SPI_eval that contain positive examples (1/ folders)
    subdirs = [entry.path for entry in os.scandir(root) 
        if entry.is_dir() and '1' in os.listdir(entry)]

    pos_dir=[]
    for subdir in subdirs:
        folders = [entry.path for entry in os.scandir(subdir) if entry.is_dir() ]
        for folder in folders:
            pos_dir.append(folder) 

    #Create ImageFolder for positive example/true_seg pair
    pass
    

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
    test_set = classification_test_set(root='./SPI_eval/', transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Evaluate
    train_or_eval(model, test_loader)

if __name__ == '__main__':
    main()
