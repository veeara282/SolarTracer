import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    '''Implements a depthwise separable 2D convolution as described
    in MobileNet (https://arxiv.org/abs/1704.04861)'''

    def __init__(self, in_channels, out_channels, kernel_size):
        # Apply the same filter to every channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)
        # Combine channels using a 1*1 kernel
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1))
