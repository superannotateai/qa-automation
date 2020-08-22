import numpy as np
import math
import torch
from torch.nn import functional as F


def add_salt_and_pepper(image_tensor, ratio = 0.97):
    #TODO: Add discriptive doc string 
    x = image_tensor.detach().clone()
    h, w = x.shape[1:] 
    salt_mask = np.random.choice((0, 1, 2), size=(1,h,w), p=[ratio, (1 - ratio) / 2., (1 - ratio) / 2.])
    salt_mask = np.repeat(salt_mask, 3, axis=0)

    x[torch.from_numpy(salt_mask == 1)] = 255
    x[torch.from_numpy(salt_mask == 2)] = 0

    return x

def gaussian_blur(image_tensor, size = 5):
    #TODO: Add discriptive doc string 
    x = image_tensor.detach().clone()
    kernel = _gaussian_kernel(size=size)
    kernel_size = 2*size + 1

    x = x[None,...]
    padding = int((kernel_size - 1) / 2)
    x = x.to(torch.float32)
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    x = torch.squeeze(F.conv2d(x, kernel, groups=3))
    x = x.to(torch.uint8)

    return x

def change_contrast(image_tensor, brightness = 0.4, contrast = 0.4):
    #TODO: Add discriptive doc string 
    x = image_tensor.detach().clone()
    x = x * contrast + brightness
    x = x.to(torch.uint8)
    return x


def _gaussian_kernel(size, sigma=2., dim=2, channels=3):
    #TODO: Add discriptive doc string 
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

