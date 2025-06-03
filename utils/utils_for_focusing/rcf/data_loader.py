from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2


def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


def is_grayscale(img):
    # 判断图像是否为灰度图
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)

def convert_to_rgb(img):
    # 将灰度图转换为RGB
    if is_grayscale(img):
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
    return img


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root=r'E:\DongJiayao\Data\PASCAL', split='train', transform=False, k_size=10):
        self.root = root
        self.split = split
        self.transform = transform
        self.k_size = k_size

        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
            with open(self.filelist, 'r') as f:
                self.filelist = f.readlines()
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
            with open(self.filelist, 'r') as f:
                self.filelist = f.readlines()
        elif self.split == 'predict':
            self.namelist = os.listdir(root)
            self.filelist = self.namelist
        else:
            raise ValueError("Invalid split type!")

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            try:
                img_file, lb_file = self.filelist[index].split()
            except ValueError:
                print(f"Error parsing line {index}: {self.filelist[index]}")
                # img_file = "aug_data/0.0_0/2010_005297.jpg"  # 替换为默认路径或跳过
                # lb_file = "aug_gt/0.0_0/2010_005297.png"
                img_file = "gray/20240429_48_2.png"  # 替换为默认路径或跳过
                lb_file = "edge/20240429_48_2.png"
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<128)] = 2
            lb[lb >= 128] = 1
            
        elif self.split == 'test':
            # img_file, _ = self.filelist[index].rstrip()
            img_file, _ = self.filelist[index].split()

        else:
            pass

        if self.split == "train":
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = convert_to_rgb(img)  # 确保图像是RGB格式
            img = prepare_image_cv2(img)
            return img, lb
        elif self.split == 'test':
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = convert_to_rgb(img)  # 确保图像是RGB格式
            img = prepare_image_PIL(img)
            return img
        else:
            img = np.array(Image.open(join(self.root, self.namelist[index])), dtype=np.float32)
            img = convert_to_rgb(img)  # 确保图像是RGB格式
            # img = img.resize((img.width // 10, img.height // 10), Image.Resampling.LANCZOS)  # 缩小四倍
            img = cv2.resize(img, (img.shape[1] // self.k_size, img.shape[0] // self.k_size), interpolation=cv2.INTER_NEAREST)
            img = prepare_image_PIL(img)
            return img, self.namelist[index]
