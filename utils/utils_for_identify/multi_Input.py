import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import os
import numpy as np
import cv2
import albumentations as A
from PIL import Image


class FourChannelToTensor(object):
    def __call__(self, pic):
        # 将 numpy 数组 (H, W, C) 转换为 torch.Tensor (C, H, W)
        if len(pic.shape) == 2:  # 如果是单通道图像
            pic = pic[:, :, np.newaxis]
        img = torch.from_numpy(pic).permute(2, 0, 1).float()
        return img


class MultiImageDataset(Dataset):
    def __init__(self, root_dir, augmentation=None, transform=None):
        self.augmentation = augmentation
        self.transform = transform
        self.root_dir = root_dir

        self.types = os.listdir(self.root_dir)  # 获取图像的维度类型
        self.classes = os.listdir(os.path.join(self.root_dir, self.types[0]))  # 获取类别目录
        self.images = []  # 用于存储所有图像的路径和标签

        # todo 修改输入图像类型
        relevant_types = {'hologram'}

        # 遍历类别目录
        for type_idx, type_name in enumerate(self.types):
            # 如果图像类型为relevant_types里面的类型
            if type_name in relevant_types:
                type_dir = os.path.join(root_dir, type_name)

                for class_idx, class_name in enumerate(os.listdir(type_dir)):
                    class_dir = os.path.join(type_dir, class_name)  # 每个type的图像名称要一模一样
                    image_files = os.listdir(class_dir)

                    # 将每个图像的路径和对应的类别索引(class_idx)存入 images 列表
                    for img_file in image_files:
                        self.images.append((img_file, class_idx))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像及其类别标签
        img_name, label = self.images[idx]

        # 加载图像 todo: 4个img分别代表4种图像，要更改输入，需要修改以下部分
        # x = os.path.join(self.root_dir, self.types[0], self.classes[label], img_name)
        # y = os.path.join(self.root_dir, self.types[1], self.classes[label], img_name)
        # z = os.path.join(self.root_dir, self.types[2], self.classes[label], img_name)
        # w = os.path.join(self.root_dir, self.types[3], self.classes[label], img_name)
        img0 = cv2.imread(os.path.join(self.root_dir, 'hologram', self.classes[label], img_name), 0)
        # img1 = cv2.imread(os.path.join(self.root_dir, 'dolp', self.classes[label], img_name), 0)
        # img2 = cv2.imread(os.path.join(self.root_dir, 'aop', self.classes[label], img_name), 0)
        # img3 = cv2.imread(os.path.join(self.root_dir, self.types[3], self.classes[label], img_name), 0)

        # todo: 4个img分别代表4种图像，要更改输入，需要修改以下部分
        img0 = img0[:, :, np.newaxis]
        # img1 = img1[:, :, np.newaxis]
        # img2 = img2[:, :, np.newaxis]
        # img3 = img3[:, :, np.newaxis]

        # todo: 4个img分别代表4种图像，要更改输入，需要修改以下部分
        # merged_array = np.concatenate((img0, img1), axis=2)

        if self.transform:
            merged_array = self.transform(img0)

        # 返回转换后的图像张量
        return merged_array, label


def get_augmentation():
    transforms = [
          A.HorizontalFlip(p=0.5),
          A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
          A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3, p=0.2),
          A.OneOf([
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
          A.OneOf([
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
          A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
      ]

    return A.Compose(transforms)


def get_torch_transforms_1channel(img_size=224):
    data_transforms = {
        'train': transforms.Compose([
            FourChannelToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),  # 添加 antialias=True
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation((-5, 5)),
            # transforms.RandomAutocontrast(p=0.2),  # 移除 autocontrast
            transforms.Normalize([0.485]*1, [0.229]*1)  # 调整为 4 通道  todo：如果不是4通道了，4就改成相应通道数
        ]),
        'val': transforms.Compose([
            FourChannelToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),  # 添加 antialias=True
            transforms.Normalize([0.485]*1, [0.229]*1)  # 调整为 4 通道  todo：如果不是4通道了，4就改成相应通道数
        ]),
    }
    return data_transforms



# data_transforms = get_torch_transforms_4channel(img_size=224)  # 获取图像预处理方式
# train_transforms = data_transforms['train']  # 训练集数据处理方式
#
# # 实例化数据集
# dataset = MultiImageDataset(
#     r'F:\Algorithm\2023_pytorch110_classification_42-master-main\data\Weather Resized_split1\test',
#     transform=get_torch_transforms_4channel(img_size=224)['train'])
#
# x = dataset[0]

# 实例化DataLoader
# train_loader = DataLoader(  # 按照批次加载训练集
#         dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True,
#     )

