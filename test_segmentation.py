import torch
from torch._C import dtype
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from tqdm.std import tqdm

from .segmentation import EfficientNetSegmentation
from utils import transform, target_transform, train_or_eval, eval_segmentation
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

class SegmentationTestSet(Dataset):
    def __init__(self, root, transform = transform, target_transform = target_transform):
        self.root = root
        # TODO create an index of (image, mask) file path pairs
        self.samples = []
        for subdir in [entry.path for entry in os.scandir(root) if entry.is_dir() and '1' in os.listdir(entry)]:
            #iterate through only positive folders (/1)
            pos_dirs = [entry.path for entry in os.scandir(subdir) if entry.is_dir() and entry.name == '1']
            for pos_dir in pos_dirs:
                valid_imgs = [entry.path for entry in os.scandir(pos_dir) if re.search(r'(\d+)\.\w+', entry.path)]
                for img in valid_imgs:
                    stem, _ = os.path.splitext(img)
                    self.samples.append((img, f"{stem}_true_seg.png"))
        self.transform = transform
        self.target_transform = target_transform
                
    def __getitem__(self, index: int):
        img_path, mask_path = self.samples[index]
        img, mask = Image.open(img_path), Image.open(mask_path)
        img = self.transform(img.convert('RGB'))
        mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.samples)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the model on the test set')
    parser.add_argument('model', metavar='model.pt', default='model.pt')
    parser.add_argument('-b', '--batch-size', type=int, default=48)
    return parser.parse_args()

def main():
    args = parse_args()

    # Read model file
    model = EfficientNetSegmentation.from_save_file(args.model)
    model.eval()

    # Set up test set
    test_set_class = classification_test_set(root='./SPI_eval/', transform=transform)
    test_loader_class = DataLoader(test_set_class, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_set_seg = SegmentationTestSet(root='./SPI_eval/')
    test_loader_seg = DataLoader(test_set_seg, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Evaluate
    train_or_eval(model, test_loader_class)
    eval_segmentation(model, test_loader_seg)

if __name__ == '__main__':
    main()
