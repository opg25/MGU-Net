import numbers
import random
import numpy as np
from PIL import Image, ImageOps
import torch

class Label_Transform(object):
    def __init__(self):
        # Define the mapping of pixel values to class indices
        self.label_ranges = [
            (0, 0),      # background: 0
            (1, 25),     # class 1
            (26, 50),    # class 2
            (51, 75),    # class 3
            (76, 100),   # class 4
            (101, 125),  # class 5
            (126, 150),  # class 6
            (151, 175),  # class 7
            (176, 200),  # class 8
            (201, 225),  # class 9
            (226, 255)   # class 10
        ]

    def __call__(self, image, label, *args):
        label = np.array(label)
        label_out = np.zeros_like(label)
        
        # Map pixel value ranges to class indices
        for class_idx, (min_val, max_val) in enumerate(self.label_ranges):
            label_out[(label >= min_val) & (label <= max_val)] = class_idx
            
        return image, Image.fromarray(label_out)

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, label=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        if label is None:
            return image,
        else:
            return image, label


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label=None):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic)
        else:
            # Convert to numpy array
            img = np.array(pic)
            
            # Ensure grayscale images are properly shaped
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            
            # Convert to tensor
            img = torch.from_numpy(img.transpose((2, 0, 1)))

        # Convert to float and normalize to [0,1]
        img = img.float().div(255)

        if label is None:
            return img,
        else:
            if isinstance(label, Image.Image):
                label = np.array(label, dtype=np.int64)
            return img, torch.from_numpy(label)


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
