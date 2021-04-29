from segmentation import get_device
import numpy as np
import torch
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms.functional import _is_numpy, _is_numpy_image

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
    if isinstance(tensor, torch.ByteTensor):
        return tensor.to(get_device()).to(torch.get_default_dtype()).div(255)
    else:
        return tensor.to(get_device())
