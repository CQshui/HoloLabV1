#!/user/bin/python
# coding=utf-8
"""
用于预测VOC数据集的边缘并输出CSV，多线程并行计算
"""
import os
import re
import numpy as np
from PIL import Image
import cv2
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
from concurrent.futures import ThreadPoolExecutor  # 新增多线程库

matplotlib.use('Agg')

try:
    from rcf.data_loader import BSDS_RCFLoader
    from rcf.models import RCF
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
parser.add_argument('--dataset', help='parent folder of subfolders',
                    default=r'F:\dongjiayao\Data\VOC\angular')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def process_subfolder(model, sub_dir):
    predict_dataset = BSDS_RCFLoader(root=sub_dir, split="predict", k_size=1)
    # 增加数据加载并行度
    predict_loader = DataLoader(
        predict_dataset, batch_size=args.batch_size,
        num_workers=16,  # 调整数据加载并行数
        drop_last=True, shuffle=False)

    concentrations = []
    names = []

    model.eval()
    with torch.no_grad():
        for idx, (image, name) in enumerate(predict_loader):
            image = image.cuda()
            _, _, H, W = image.shape
            results = model(image)
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            result_image = Image.fromarray((result * 255).astype(np.uint8))
            concentration = calculate_brightness_concentration(np.array(result_image))
            names.extend(name)
            concentrations.append(concentration)
            print(f"Processed {name[0]} in {os.path.basename(sub_dir)}")

    sorted_indices = sorted(range(len(names)), key=lambda x: int(re.search(r'\d+', names[x]).group()))
    sorted_concentrations = [concentrations[i] for i in sorted_indices]
    return (os.path.basename(sub_dir), sorted_concentrations)


def calculate_brightness_concentration(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    total_brightness = image.sum()
    nonzero_pixels = np.count_nonzero(image)
    return total_brightness / (nonzero_pixels * image.shape[0] * image.shape[1]) if nonzero_pixels != 0 else 0  # todo:有修改


def main():
    args.cuda = True
    model = RCF()
    model.cuda()

    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("Checkpoint loaded successfully.")

    parent_dir = args.dataset
    sub_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))]

    results = {}

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=16) as executor:  # 根据GPU显存调整线程数
        futures = []
        for sub_dir in sub_dirs:
            # 每个任务提交到线程池
            futures.append(executor.submit(process_subfolder, model, sub_dir))

        # 获取处理结果
        for future in futures:
            folder_name, concentrations = future.result()
            results[folder_name] = concentrations
            print(f"Completed processing {folder_name}")

    # 创建DataFrame并保存
    df = pd.DataFrame({k: pd.Series(v) for k, v in results.items()})
    os.makedirs(args.tmp, exist_ok=True)
    csv_path = os.path.join(args.tmp, 'concentration_results1.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    main()
