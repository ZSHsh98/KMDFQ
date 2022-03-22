import torch
from torchvision import  transforms
import numpy as np
import os
from PIL import Image

__all__ = ["save_tensor_as_image"]

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tensor_to_image(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def save_tensor_as_image(tensor, filename, path='save_images/generated'):
    """
    save the tensor ad image
    :params tensor: target tensor
    :params filename: filename to use
    :params path: the file path to save
    """
    mkdirs(path)
    image = tensor_to_image(tensor)
    image.save(os.path.join(path, filename))






