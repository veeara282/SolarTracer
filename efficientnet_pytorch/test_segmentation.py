import torch
from torch._C import dtype
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from tqdm.std import tqdm

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

def threshold(image_mask: torch.FloatTensor, threshold=0.6) -> torch.ByteTensor:
    '''Converts a FloatTensor to a binary LongTensor using a threshold.'''
    return (image_mask > threshold).to(torch.uint8)

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

def sum_2d(mask):
    return torch.sum(mask, dim=(1, 2), dtype=torch.int32)

def divide_and_sum(quantity1, quantity2):
    return torch.sum(quantity1 / quantity2).cpu().item()

def eval_segmentation(model: torch.nn.Module, data_loader: DataLoader):
    model.eval()
    total_jaccard = 0.0
    total_precision = 0.0
    total_recall = 0.0
    example_count = 0
    for batch in tqdm(data_loader, desc="Test segmentation"):
        # These are tensors of input images and segmentation masks respectively
        inputs, masks = batch[0], torch.squeeze(batch[1], 1)
        output_cams = model(to_device(inputs), return_cam=True)
        # Compare model output and true segmentation mask on GPU
        outputs = threshold(output_cams)
        masks = to_device(masks)
        # Compute Jaccard similarity, precision, and recall for each example
        intersection = torch.logical_and(outputs, masks).to(torch.uint8)
        union = torch.logical_or(outputs, masks).to(torch.uint8)
        count_intersection = sum_2d(intersection)
        count_union = sum_2d(union)
        count_pred = sum_2d(outputs)
        count_gold = sum_2d(masks)
        # Sum up Jaccard similarity, precision, and recall
        total_jaccard += divide_and_sum(count_intersection, count_union)
        total_precision += divide_and_sum(count_intersection, count_pred)
        total_recall += divide_and_sum(count_intersection, count_gold)
        example_count += inputs.shape[0]
    avg_jaccard = total_jaccard / example_count
    avg_precision = total_precision / example_count
    avg_recall = total_recall / example_count
    print('Segmentation results:')
    print(f'Average Jaccard similarity: {avg_jaccard:.2%}')
    print(f'Average precision: {avg_precision:.2%}')
    print(f'Average recall: {avg_recall:.2%}')
    print(f'Number of examples: {example_count}')


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
    test_set_class = classification_test_set(root='./SPI_eval/', transform=transform)
    test_loader_class = DataLoader(test_set_class, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_set_seg = SegmentationTestSet(root='./SPI_eval/')
    test_loader_seg = DataLoader(test_set_seg, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Evaluate
    train_or_eval(model, test_loader_class)
    eval_segmentation(model, test_loader_seg)

if __name__ == '__main__':
    main()
