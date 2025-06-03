#!/user/bin/python
# coding=utf-8
import os
import re
import numpy as np
from PIL import Image
import cv2
import argparse
import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')

try:
    from data_loader import BSDS_RCFLoader
    from models import RCF
except:
    pass

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default=r'F:\dongjiayao\Pycharm\Holo-Track\rcf\trained_model\checkpoint_epoch29.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/RCF')
# ================ dataset
# parser.add_argument('--dataset', help='root folder of dataset', default=r'E:\DongJiayao\Data\PASCAL')
parser.add_argument('--dataset', help='root folder of dataset',
                    default=r'F:\dongjiayao\Data\VOC\angular\2011_003246')
args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():
    args.cuda = True
    # dataset
    predict_dataset = BSDS_RCFLoader(root=args.dataset, split="predict", k_size=1)
    predict_loader = DataLoader(
        predict_dataset, batch_size=args.batch_size,
        num_workers=1, drop_last=True, shuffle=True)

    # model
    model = RCF()
    model.cuda()

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'"
          .format(args.resume))

    rcf_predict(model, predict_loader, save_dir=r'F:\dongjiayao\Data\VOC\tmp')


def rcf_predict(model, predict_loader, save_dir=''):
    model.eval()

    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # 存储文件名和对应的熵值
    names = []
    concentrations = []
    max_concentration = -1
    best_focus_image = None
    best_focus_name = None

    for idx, (image, name) in enumerate(predict_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        result = Image.fromarray((result * 255).astype(np.uint8))
        # 边缘检测图像保存
        result.save(os.path.join(save_dir, name[0]))

        # # 计算熵值
        # if idx == 1:
        #     result = np.zeros_like(np.array(result))

        # entropy = calculate_entropy(np.array(result))
        concentration = calculate_brightness_concentration(np.array(result))
        names.append(name[0])  # 提取文件名
        concentrations.append(concentration)

        # 更新最大 concentration 的图像
        if concentration > max_concentration:
            max_concentration = concentration
            best_focus_image = image
            best_focus_name = name

        print("Running test [%d/%d]" % (idx + 1, len(predict_loader)))
        print(name, concentration)

    # 按文件名中的编号排序
    # 假设文件名格式为 image_{number}.jpg
    # 使用正则表达式提取编号
    sorted_indices = sorted(range(len(names)), key=lambda x: int(re.findall(r'\d+', names[x])[0]))
    sorted_names = [names[i] for i in sorted_indices]
    sorted_concentrations = [concentrations[i] for i in sorted_indices]

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_names, sorted_concentrations, marker='o', linestyle='-', color='b')
    plt.xlabel('Image Name (Sorted by Number)')
    plt.ylabel('Concentration')
    plt.title('Concentration of Predicted Images')
    plt.xticks(rotation=45)  # 旋转横坐标标签
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局

    if save_dir:
        # 保存折线图
        plot_path = os.path.join(save_dir, 'concentration_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Concentration plot saved to {plot_path}")

    return best_focus_image, best_focus_name


def calculate_entropy(image):
    # 计算灰度直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # 归一化为概率
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # 避免对 0 取对数
    return entropy

# def calculate_brightness_concentration(image):
#     brightness = image.sum()
#     concentration = (brightness ** 5).sum() / brightness.sum()
#     return concentration


def calculate_brightness_concentration(image):
    # 确保输入为NumPy数组
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)

    # 确保为灰度图像
    if len(image.shape) == 3:  # 彩色图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算亮度总和与非零像素个数
    total_brightness = image.sum()
    nonzero_pixels = np.count_nonzero(image)

    # 避免除以零
    if nonzero_pixels == 0:
        return 0

    # 计算集中度
    concentration = total_brightness / nonzero_pixels

    # 计算亮度方差（无偏估计，ddof=1）
    non_zero_values = image[image != 0]
    variance = np.var(non_zero_values, ddof=1)

    return variance


if __name__ == '__main__':
    main()
