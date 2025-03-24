from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RandomFlip:
    """Randomly flips an image and its mask or bounding boxes horizontally and/or vertically."""

    def __init__(self, ph: float, pv: float):
        assert isinstance(ph, float), "ph should be a float"
        assert isinstance(pv, float), "pv should be a float"

        self.ph = ph  # Probability of horizontal flip
        self.pv = pv  # Probability of vertical flip

    def __call__(
        self, sample: Tuple[torch.Tensor, Union[Dict, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Union[Dict, torch.Tensor]]:
        image, target = sample

        if isinstance(target, dict):  # Bounding boxes case
            boxes = target["boxes"]
            height, width = image.shape[-2:]

            if np.random.random() < self.ph:  # Horizontal flip
                image = TF.hflip(image)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x1, x2

            if np.random.random() < self.pv:  # Vertical flip
                image = TF.vflip(image)
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]  # Flip y1, y2

            target["boxes"] = boxes  # Update bounding boxes

        else:  # Mask case
            if np.random.random() < self.ph:
                image = TF.hflip(image)
                target = TF.hflip(target)

            if np.random.random() < self.pv:
                image = TF.vflip(image)
                target = TF.vflip(target)

        return image, target


from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF


def zoom(image: torch.Tensor, zoom_factor: float) -> torch.Tensor:
    """Applies a zoom effect by cropping and resizing the image with a random crop."""
    height, width = image.shape[-2:]
    # Apply random zoom factor
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Random crop position
    top = np.random.randint(0, height - new_height)  # Random top position
    left = np.random.randint(0, width - new_width)  # Random left position

    image = TF.crop(image, top, left, new_height, new_width)
    image = TF.resize(image, (height, width))

    return image, (top, left, new_width, new_height)


class RandomZoom:
    """Applies a random zoom transformation to an image and its mask or bounding boxes."""

    def __init__(self, zoom_factor_range: Tuple[float, float], p: float):
        assert (
            isinstance(zoom_factor_range, tuple) and len(zoom_factor_range) == 2
        ), "zoom_factor_range should be a tuple of two floats"
        assert isinstance(p, float), "p should be a float"

        self.zoom_factor_range = zoom_factor_range
        self.p = p

    def __call__(
        self, sample: Tuple[torch.Tensor, Union[Dict, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Union[Dict, torch.Tensor]]:
        image, target = sample

        if np.random.random() < self.p:
            # Sample a random zoom factor from the specified range
            zoom_factor = np.random.uniform(
                self.zoom_factor_range[0], self.zoom_factor_range[1]
            )

            image, (top, left, new_width, new_height) = zoom(image, zoom_factor)

            if isinstance(target, dict):  # Adjust bounding boxes
                boxes = target["boxes"]
                height, width = image.shape[-2:]
                scale_x = width / new_width
                scale_y = height / new_height

                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) * scale_x
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) * scale_y

                target["boxes"] = boxes  # Update bounding boxes

            else:  # Mask case
                target = zoom(target, zoom_factor)[0]  # Apply zoom

        return image, target


class Resize:
    """Resizes an image and its mask or bounding boxes to a given output size."""

    def __init__(self, output_size: Tuple[int, int]):
        assert (
            isinstance(output_size, tuple) and len(output_size) == 2
        ), "output_size should be a tuple (height, width)"
        self.output_size = output_size
        self.resize = transforms.Resize(output_size)

    def __call__(
        self, sample: Tuple[torch.Tensor, Union[Dict, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Union[Dict, torch.Tensor]]:
        image, target = sample

        height, width = image.shape[-2:]
        orig_size = torch.tensor([width, height], dtype=torch.float32)

        image = self.resize(image)
        new_width, new_height = self.output_size
        new_size = torch.tensor([new_width, new_height], dtype=torch.float32)
        scale = new_size / orig_size  # Scaling factors (w_scale, h_scale)

        if isinstance(target, dict):  # Adjust bounding boxes
            boxes = target["boxes"]
            boxes[:, [0, 2]] *= scale[0]  # Scale x-coordinates
            boxes[:, [1, 3]] *= scale[1]  # Scale y-coordinates
            target["boxes"] = boxes  # Update bounding boxes

        else:  # Mask case
            target = self.resize(target)

        return image, target


class ToTensor:
    """Converts an image and its mask or bounding boxes to PyTorch tensors."""

    def __call__(
        self, sample: Tuple[torch.Tensor, Union[Dict, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Union[Dict, torch.Tensor]]:
        image, target = sample
        image = TF.to_tensor(image).to(torch.float32)  # Convert image to float tensor

        if isinstance(target, dict):  # Bounding boxes case
            target["boxes"] = target["boxes"].to(torch.float32)
            target["labels"] = target["labels"].to(torch.int64)
        else:  # Mask case
            target = TF.to_tensor(target).to(
                torch.uint8
            )  # Convert mask to uint8 tensor

        return image, target
