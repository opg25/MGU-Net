from torch.utils.data import Dataset
from os.path import join,exists
from PIL import Image, ImageOps
import torch
import os
import os.path as osp
import numpy as np 
import torchvision.transforms as tt
import data.seg_transforms as st
import PIL
import random


class segList(Dataset):
    def __init__(self, data_dir, phase, transforms):
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.read_lists()

    def __getitem__(self, index):
        try:
            if self.phase == 'train':
                self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
                self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
                data = [self.load_image(self.image_list[index])]
                data.append(self.load_image(self.label_list[index]))
                data = list(self.transforms(*data))
                data = [data[0], data[1].long()]
                return tuple(data)

            if self.phase in ['eval', 'test']:
                self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
                self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
                data = [self.load_image(self.image_list[index])]
                imt = torch.from_numpy(np.array(data[0]))
                data.append(self.load_image(self.label_list[index]))
                data = list(self.transforms(*data))
                image, label = data[0], data[1]
                imn = os.path.basename(self.image_list[index])
                return image, label.long(), imt, imn

            if self.phase == 'predict':
                self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
                data = [self.load_image(self.image_list[index])]
                imt = torch.from_numpy(np.array(data[0]))
                data = list(self.transforms(*data))
                image = data[0]
                imn = os.path.basename(self.image_list[index])
                return image, imt, imn
        
        except (PIL.UnidentifiedImageError, ValueError, IOError) as e:
            print(f"Skipping corrupted image: {self.image_list[index]} - {e}")
            return None  # or some default value

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):    
        self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
        print(f'Total amount of {self.phase} images: {len(self.image_list)}')

    def load_image(self, filepath):
        try:
            target_size = (480, 480)  # Set fixed dimensions that match network expectations
            
            if filepath.endswith(".npy"):
                arr = np.load(filepath)
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = ((arr - arr.min()) * (255.0 / (arr.max() - arr.min()))).astype(np.uint8)
                img = Image.fromarray(arr)
            else:
                img = Image.open(filepath)

            # Resize all images to target size
            if 'mask' in filepath:
                # Use nearest neighbor for masks to preserve label values
                img = img.resize(target_size, Image.NEAREST)
            else:
                # Convert input images to grayscale and resize
                if img.mode != 'L':
                    img = img.convert('L')
                img = img.resize(target_size, Image.BILINEAR)
            
            return img

        except Exception as e:
            print(f"Error loading image {filepath}: {str(e)}")
            raise


def get_list_dir(phase, type, data_dir):
    data_dir = os.path.join(data_dir, phase, type)
    return [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
