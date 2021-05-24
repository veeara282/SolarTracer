import os
import random
import re

import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms.functional import _is_numpy, _is_numpy_image
from tqdm import tqdm

def to_device(obj):
    return obj.to("cuda:0" if torch.cuda.is_available() else "cpu")

def random_rotate_reflect_90(pic: Image):
    """Randomly applies one of the following transformations:
    - `None`: leaves the image as is.
    - `Image.ROTATE_90`, `Image.ROTATE_180`, `Image.ROTATE_270`: rotates by 90, 180, or 270 degrees respectively.
    - `Image.FLIP_LEFT_RIGHT`: reflects the image over the horizontal axis.
    - `Image.FLIP_TOP_BOTTOM`: reflects the image over the vertical axis.
    - `Image.TRANSPOSE`, `Image.TRANSVERSE`: reflects the image over a diagonal line ±45 degrees from the horizontal.
    
    It can be shown that any two of these transformations composed together, or one of them repeated, is equivalent to
    a single transformation of this set. Mathematically:
    - Two rotations by multiples of 90 degrees = rotation by a multiple of 90 degrees.
    - Two reflections by multiples of 45 degrees = rotation by a multiple of 90 degrees.
    - A rotation by a multiple of 90 degrees and a reflection by a multiple of 45 degrees, in either order = 
    reflection by a multiple of 45 degrees.
    
    Source: https://en.wikipedia.org/wiki/Rotations_and_reflections_in_two_dimensions

    This means that we can rotate and flip the image simultaneously with a single transformation instead of doing separate
    horizontal and vertical flips.
    """
    transform = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
                            Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE, Image.TRANSVERSE])
    if transform is not None:
        return pic.transpose(transform)
    else:
        return pic

def threshold(image_mask: torch.FloatTensor, threshold=0.6) -> torch.ByteTensor:
    '''Converts a FloatTensor to a binary LongTensor using a threshold.'''
    return (image_mask > threshold).to(torch.uint8)

'''Utility functions for reducing the data transfer bottleneck.

Idea: move image tensors to the GPU *before* converting them to the default floating-point type,
to reduce the overhead from data transfer.'''

def to_byte_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor (but don't auto-convert from int8 to float).
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # change: don't convert to FloatTensor if it's a ByteTensor
        return img

    # change: don't handle accimage.Image

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    # change: don't convert to FloatTensor if it's a ByteTensor
    return img

def to_float_tensor_gpu(tensor):
    '''Move the given tensor to the GPU, then convert it to float on the GPU, to reduce data transfer.'''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(tensor, torch.ByteTensor):
        return tensor.to(device).to(torch.get_default_dtype()).div(255)
    else:
        return tensor.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(to_byte_tensor)
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(random_rotate_reflect_90),
    transforms.Lambda(to_byte_tensor)
])

# The transformation applied to image segmentation masks.
# Image masks are 8-bit grayscale PNG images (mode L), so transforms.Grayscale is a no-op.
target_transform = transforms.Compose([
    # no-op, but just in case the image format is different from expected
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Lambda(threshold)
])

class SegmentationDataset(Dataset):
    def __init__(self, root, transform = transform, target_transform = target_transform):
        # create an index of (image, mask) file path pairs
        self.samples = []
        #iterate through only positive folder (/1)
        pos_dir = os.path.join(root, '1')
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

def train_or_eval(model: nn.Module,
                  data_loader: DataLoader,
                  optimizer: optim.Optimizer = None,
                  scaler: GradScaler = None):
    if optimizer:
        model.train()
    else:
        model.eval()
    total = 0
    total_loss = 0
    correct = 0
    true_pos, all_true, all_pos = 0, 0, 0
    for batch in tqdm(data_loader, desc=("Training Batches" if optimizer else "Validation Batches")):
        inputs, labels = batch[0], batch[1]
        # Reset the optimizer first
        if optimizer:
            optimizer.zero_grad()
        # Do both autocast steps together
        with autocast(scaler is not None):
            inputs_float = to_float_tensor_gpu(inputs)
            output = model(inputs_float)
            loss = model.loss_criterion(output, to_device(labels))
        # Accumulate metrics
        total_loss += loss.float().item()
        total += output.size()[0]
        predicted = torch.argmax(output, 1).cpu()
        correct += (labels == predicted).numpy().sum()
        true_pos += (labels & predicted).numpy().sum()
        all_true += labels.numpy().sum()
        all_pos += predicted.numpy().sum()
        # Training step
        if optimizer:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    # Compute metrics
    total_loss /= total
    precision = true_pos / all_pos
    recall = true_pos / all_true
    f1 = 2 * precision * recall / (precision + recall)
    # Print accuracy
    print('Training Results:' if optimizer else 'Validation Results:')
    print(f'Loss: {total_loss:.4f}')
    print(f'Correct: {correct} ({correct / total:.2%})')
    print(f'Precision: {true_pos} / {all_pos} ({precision:.2%})')
    print(f'Recall: {true_pos} / {all_true} ({recall:.2%})')
    print(f'F1: {f1:.2%}')
    print(f'Total: {total}\n')
    return {
        'loss': total_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
