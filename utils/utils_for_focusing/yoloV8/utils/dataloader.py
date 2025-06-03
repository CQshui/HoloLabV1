from random import sample, shuffle
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from yoloV8.utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, input_shape, num_classes, epoch_length,
                 mosaic=True, mixup=True, mosaic_prob=0.5, mixup_prob=0.5, train=True, special_aug_ratio=0.7):
        """
        Args:
            images_dir (str): 图像文件夹路径。
            labels_dir (str): 标签文件夹路径。
            input_shape (list or tuple): 输入图像的尺寸 (height, width)。
            num_classes (int): 类别数量。
            epoch_length (int): 训练的总 epoch 数。
            mosaic (bool): 是否使用 Mosaic 数据增强。
            mixup (bool): 是否使用 Mixup 数据增强。
            mosaic_prob (float): 使用 Mosaic 的概率。
            mixup_prob (float): 使用 Mixup 的概率。
            train (bool): 是否为训练模式。
            special_aug_ratio (float): 特殊数据增强的比例。
        """
        super(YoloDataset, self).__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_shape = tuple(input_shape)  # 将列表转换为元组
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1

        # 获取所有图像文件的路径
        self.image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if
                            f.endswith(('.jpg', '.png', '.jpeg'))]
        self.label_files = [os.path.join(labels_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in
                            self.image_files]

        self.length = len(self.image_files)  # 数据集长度
        self.bbox_attrs = 5 + num_classes  # 每个边界框的属性数量

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            # 随机选择 3 张图像和当前图像进行 Mosaic 数据增强
            indices = sample(range(self.length), 3)
            indices.append(index)
            shuffle(indices)
            lines = [self.image_files[i] for i in indices]
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            # 如果启用 Mixup 数据增强，则进一步进行 Mixup
            if self.mixup and self.rand() < self.mixup_prob:
                mix_index = sample(range(self.length), 1)[0]
                image_2, box_2 = self.get_random_data(self.image_files[mix_index], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            # 不使用 Mosaic 数据增强时，直接加载单张图像
            image, box = self.get_random_data(self.image_files[index], self.input_shape, random=self.train)

        # 图像预处理
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        #---------------------------------------------------#
        #   对真实框进行预处理
        #---------------------------------------------------#
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到 0-1 之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   调整格式为 (中心x, 中心y, 宽, 高)
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            #---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out 中序号为 0 的部分在 collate_fn 中处理
            #---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image, labels_out

    def yolo_to_corners(self, yolo_boxes, image_size):
        """
        将 YOLO 格式的标签 (class_id, x_center, y_center, w, h) 转换为 (x_min, y_min, x_max, y_max, class_id)。
        """
        if len(yolo_boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # 获取图像的宽度和高度
        iw, ih = image_size

        # 提取中心坐标和宽高
        x_center = yolo_boxes[:, 0] * iw
        y_center = yolo_boxes[:, 1] * ih
        w = yolo_boxes[:, 2] * iw
        h = yolo_boxes[:, 3] * ih

        # 计算边界框的左上角和右下角坐标
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        x_max = x_center + w / 2
        y_max = y_center + h / 2

        # 组合成 (x_min, y_min, x_max, y_max) 格式
        corners = np.stack([x_min, y_min, x_max, y_max], axis=-1)
        return corners

    def rand(self, a=0, b=1):
        """
        生成 [a, b) 范围内的随机数。
        """
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_path, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """
        加载图像和标签，并进行数据增强。
        """
        # 加载图像
        image = Image.open(image_path)
        image = cvtColor(image)

        # 加载标签
        label_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = np.array([line.strip().split() for line in f.readlines()], dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        # 将 YOLO 格式的标签转换为 (x_min, y_min, x_max, y_max, class_id)
        if len(labels) > 0:
            labels[:, 1:5] = self.yolo_to_corners(labels[:, 1:5], image.size)
            # 将每个子列表的第一个元素（0）移动到最后
            labels = np.hstack([labels[:, 1:], labels[:, 0].reshape(-1, 1)])

        # 其余部分保持不变
        iw, ih = image.size
        h, w = input_shape

        if not random:
            # 如果不进行数据增强，则直接调整图像尺寸并填充
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 将图像多余的部分加上灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 对真实框进行调整
            if len(labels) > 0:
                np.random.shuffle(labels)
                labels[:, [0, 2]] = labels[:, [0, 2]] * nw / iw + dx
                labels[:, [1, 3]] = labels[:, [1, 3]] * nh / ih + dy
                labels[:, 0:2][labels[:, 0:2] < 0] = 0
                labels[:, 2][labels[:, 2] > w] = w
                labels[:, 3][labels[:, 3] > h] = h
                box_w = labels[:, 2] - labels[:, 0]
                box_h = labels[:, 3] - labels[:, 1]
                labels = labels[np.logical_and(box_w > 1, box_h > 1)]  # 丢弃无效框

            return image_data, labels

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 对真实框进行调整
        if len(labels) > 0:
            np.random.shuffle(labels)
            labels[:, [0, 2]] = labels[:, [0, 2]] * nw / iw + dx   # todo
            labels[:, [1, 3]] = labels[:, [1, 3]] * nh / ih + dy
            if flip:
                labels[:, [0, 2]] = w - labels[:, [2, 0]]
            labels[:, 0:2][labels[:, 0:2] < 0] = 0
            labels[:, 2][labels[:, 2] > w] = w
            labels[:, 3][labels[:, 3] > h] = h
            box_w = labels[:, 2] - labels[:, 0]
            box_h = labels[:, 3] - labels[:, 1]
            labels = labels[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, labels

    def get_random_data_with_Mosaic(self, image_paths, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        """
        Mosaic 数据增强。
        """
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for image_path in image_paths:
            # 加载图像
            image = Image.open(image_path)
            image = cvtColor(image)

            # 加载标签
            label_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = np.array([line.strip().split() for line in f.readlines()], dtype=np.float32)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

            # 将 YOLO 格式的标签转换为 (x_min, y_min, x_max, y_max, class_id)
            if len(labels) > 0:
                labels[:, 1:5] = self.yolo_to_corners(labels[:, 1:5], image.size)

            # 翻转图像
            flip = self.rand() < .5
            if flip and len(labels) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                labels[:, [0, 2]] = image.size[0] - labels[:, [2, 0]]

            # 对图像进行缩放并且进行长和宽的扭曲
            new_ar = image.size[0] / image.size[1] * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图片进行放置，分别对应四张分割图片的位置
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []

            # 对 box 进行重新处理
            if len(labels) > 0:
                np.random.shuffle(labels)
                labels[:, [0, 2]] = labels[:, [0, 2]] * nw / image.size[0] + dx
                labels[:, [1, 3]] = labels[:, [1, 3]] * nh / image.size[1] + dy
                labels[:, 0:2][labels[:, 0:2] < 0] = 0
                labels[:, 2][labels[:, 2] > w] = w
                labels[:, 3][labels[:, 3] > h] = h
                box_w = labels[:, 2] - labels[:, 0]
                box_h = labels[:, 3] - labels[:, 1]
                labels = labels[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(labels), 5))
                box_data[:len(labels)] = labels

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)

        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def merge_bboxes(self, bboxes, cutx, cuty):
        """
        合并 Mosaic 数据增强后的边界框。
        """
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        """
        Mixup 数据增强。
        """
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


# DataLoader 中 collate_fn 使用
def yolo_dataset_collate(batch):
    """
    将批次数据整理为模型输入格式。
    """
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes

# # DataLoader中collate_fn使用
# def yolo_dataset_collate(batch):
#     images      = []
#     n_max_boxes = 0
#     bs          = len(batch)
#     for i, (img, box) in enumerate(batch):
#         images.append(img)
#         n_max_boxes = max(n_max_boxes, len(box))
    
#     bboxes  = torch.zeros((bs, n_max_boxes, 4))
#     labels  = torch.zeros((bs, n_max_boxes, 1))
#     masks   = torch.zeros((bs, n_max_boxes, 1))
    
#     for i, (img, box) in enumerate(batch):
#         _sub_length = len(box)
#         bboxes[i, :_sub_length] = box[:, :4]
#         labels[i, :_sub_length] = box[:, 4]
#         masks[i, :_sub_length]  = 1
    
#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes, labels, masks
