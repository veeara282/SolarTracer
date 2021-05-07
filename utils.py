import random

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms.functional import _is_numpy, _is_numpy_image

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
