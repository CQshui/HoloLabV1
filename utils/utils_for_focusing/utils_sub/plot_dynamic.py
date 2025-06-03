import os
import numpy as np
import torch
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from final import *
from rcf.models import RCF
import matplotlib
import pywt
import csv
import random
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

matplotlib.use("TkAgg")


# 定义计算熵的函数，熵越小越聚焦
def calculate_entropy(image_array):
    gray_image = image_array
    gray_image = gray_image.astype(np.uint8)  # 确保像素值为8位整数
    hist, _ = np.histogram(gray_image.ravel(), bins=256, range=[0, 256])
    hist = hist / np.sum(hist)  # 概率分布
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # 防止log(0)
    entropy_value = -entropy
    return entropy_value


# 定义计算LAP的函数
def calculate_lap(image_array):
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    lap_result = convolve2d(image_array, laplacian, mode='valid')
    lap_value = np.sum(np.abs(lap_result))  # 平均绝对梯度
    return lap_value


# 定义计算VAR的函数
def calculate_var(image_array):
    return np.var(image_array)


# 定义计算余弦相似度（CS）的函数，越大越聚焦
def calculate_cosine_similarity(image1, image2):
    vec1 = image1.flatten()
    vec2 = image2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm1 * norm2 + 1e-10)  # 防止除以零
    return cosine_similarity


def calculate_wavelet(image_array):
    # 确保图像尺寸为偶数
    h, w = image_array.shape
    if h % 2 != 0:
        image_array = image_array[:h - 1, :]
    if w % 2 != 0:
        image_array = image_array[:, :w - 1]
    # 进行一层小波分解（使用haar小波）
    coeffs = pywt.dwt2(image_array, 'haar')
    cA, (cH, cV, cD) = coeffs
    # 计算高频细节系数的能量（平方和）
    energy = (cH ** 2 + cV ** 2 + cD ** 2).sum()
    return energy


def rcf_value(model, img, save_dir=''):
    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    img_0 = img
    img = convert_to_rgb(img)  # 确保图像是RGB格式
    img = cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10), interpolation=cv2.INTER_NEAREST)

    img = prepare_image_cv2(img)
    img = torch.from_numpy(img).float()  # 确保是浮点格式
    img = img.cuda()

    # 如果需要，重新调整维度
    if img.dim() == 3:  # 如果是单通道图像，先调整维数
        img = img.unsqueeze(0)  # 添加批量维度
        img = img.permute(0, 1, 2, 3)  # 调整为 (B, C, H, W)

    _, _, H, W = img.shape
    results = model(img)
    result = torch.squeeze(results[-1].detach()).cpu().numpy()
    result = Image.fromarray((result * 255).astype(np.uint8))

    # 边缘检测图像保存
    tmp_pth = os.path.join(r'F:\dongjiayao\Pycharm\Holo-Track\img\edge1', str(time.time()))
    os.mkdir(tmp_pth)
    result.save(os.path.join(tmp_pth, "edge.jpg"))
    # img_ori_tmp = torch.squeeze(image_origin.squeeze()).cpu().numpy()
    img_ori_tmp = img_0
    img_ori_tmp = img_ori_tmp.astype(np.uint8)
    cv2.imwrite(os.path.join(tmp_pth, "image.jpg"), img_ori_tmp)

    # 计算亮度密度
    concentration = calculate_brightness_concentration(np.array(result))

    return concentration


# 根据图像名称解析距离值
def parse_distance(image_name):
    # 示例：offaxis_28_0.0004613.jpg → 距离为0.0004613
    parts = image_name.split('_')
    if len(parts) >= 3 and parts[-1].endswith('.jpg'):
        distance_part = parts[-1].replace('.jpg', '')
        try:
            return float(distance_part)
        except:
            return None
    return None


class PlotApp:
    def __init__(self, root, image_folder, rcf_model):
        self.root = root
        self.image_folder = image_folder
        self.rcf_model = rcf_model
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.data_points = []  # 用于存储所有计算的点

        # 创建图形和画布
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 创建按钮
        self.button = ttk.Button(self.root, text="随机计算并绘制点", command=self.plot_random_point)
        self.button.pack(side=tk.BOTTOM)

        # 初始化绘图
        self.ax.set_xlabel('Distance')
        self.ax.set_ylabel('Normalized Value')
        self.ax.set_title('Comparison of Different Criteria')
        self.ax.grid(True)
        self.ax.legend()

    def plot_random_point(self):
        if not self.image_files:
            print("所有图像已处理完毕")
            return

        # 随机选择一张图像
        image_file = random.choice(self.image_files)
        self.image_files.remove(image_file)  # 从列表中移除已处理的图像

        # 解析距离值
        distance = parse_distance(image_file)
        if distance is None:
            print(f"无法解析图像 {image_file} 的距离值，跳过")
            return

        # 读取图像并计算各聚焦判据
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert("L")
        image_array = np.array(image, dtype=np.float32)

        entropy = calculate_entropy(image_array)
        lap = calculate_lap(image_array)
        var = calculate_var(image_array)
        wavelet = calculate_wavelet(image_array)
        rcf_v = rcf_value(self.rcf_model, image_array, 'F:\dongjiayao\Pycharm\Holo-Track\img\edge1')

        # 计算余弦相似度（仅当有前一张图像时）
        cosine_sim = np.nan
        if self.data_points:
            prev_array = np.array(self.data_points[-1]['image'])
            cosine_sim = calculate_cosine_similarity(prev_array, image_array)

        # 保存当前图像和数据点
        self.data_points.append({
            'distance': distance,
            'entropy': entropy,
            'lap': lap,
            'var': var,
            'cosine': cosine_sim,
            'wavelet': wavelet,
            'rcf': rcf_v,
            'image': image
        })

        # 绘制点
        self.ax.scatter(distance, rcf_v, color='purple', label='RCF' if len(self.data_points) == 1 else "")

        if len(self.data_points) > 1:
            # 按距离排序数据点
            sorted_points = sorted(self.data_points, key=lambda x: x['distance'])
            distances = [p['distance'] for p in sorted_points]
            rcf_values = [p['rcf'] for p in sorted_points]

            # 找到最大RCF值的索引
            peak_index = np.argmax(rcf_values)
            print(distances[peak_index])

            # 清空之前的折线
            for line in self.ax.lines:
                line.remove()

            # 计算斜率
            if peak_index > 0:
                k = (rcf_values[peak_index] - rcf_values[0]) / (distances[peak_index] - distances[0])
                # 绘制左半部分折线
                x_left = np.linspace(distances[0], distances[peak_index], 100)
                y_left = rcf_values[0] + k * (x_left - distances[0])
                self.ax.plot(x_left, y_left, color='purple', linestyle='-')

            if peak_index < len(distances) - 1:
                # 绘制右半部分折线
                x_right = np.linspace(distances[peak_index], distances[-1], 100)
                y_right = rcf_values[peak_index] - k * (x_right - distances[peak_index])
                self.ax.plot(x_right, y_right, color='purple', linestyle='-')

        # 更新画布
        self.canvas.draw()


if __name__ == "__main__":
    # 初始化RCF模型
    rcf_model = RCF()
    rcf_model.cuda()
    rcf_model.eval()
    checkpoint = torch.load(r'F:\dongjiayao\Pycharm\Holo-Track\rcf\trained_model\checkpoint_epoch29.pth', weights_only=True)
    rcf_model.load_state_dict(checkpoint['state_dict'])

    # 图像文件夹路径
    image_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\1\cut1"  # 实际颗粒

    # 创建Tkinter窗口
    root = tk.Tk()
    root.title("聚焦值对比")
    app = PlotApp(root, image_folder, rcf_model)
    root.mainloop()