import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class RandomFlip:
    def __init__(self, ph, pv):
        assert isinstance(ph, float)
        assert isinstance(pv, float)

        self.ph = ph
        self.pv = pv

    def __call__(self, sample):
        image, mask = sample

        if np.random.random() < self.ph:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if np.random.random() < self.pv:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return (image, mask)


def zoom(image, zoom_factor):
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    top = (height - new_height) // 2
    left = (width - new_width) // 2

    image = TF.crop(image, top, left, new_height, new_width)
    image = TF.resize(image, (height, width))

    return image


class RandomZoom:
    def __init__(self, zoom_factor, p):
        assert isinstance(zoom_factor, (int, float))
        assert isinstance(p, float)
        self.zoom_factor = zoom_factor
        self.p = p

    def __call__(self, sample):
        image, mask = sample

        if np.random.random() < self.p:
            image = zoom(image, self.zoom_factor)
            mask = zoom(mask, self.zoom_factor)

        return (image, mask)


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.resize = transforms.Resize(output_size)

    def __call__(self, sample):
        image, mask = sample

        image = self.resize(image)
        mask = self.resize(mask)

        return (image, mask)


class ToTensor:
    def __call__(self, sample):
        image, mask = sample

        image = TF.to_tensor(image).to(torch.uint8)
        mask = TF.to_tensor(mask).to(torch.uint8)

        return (image, mask)
