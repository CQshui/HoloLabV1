"""
用于绘制RCF对比其他聚焦判据的曲线图
"""
import os
import numpy as np
import torch
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from final import *
from rcf.models import RCF
import matplotlib

matplotlib.use("TkAgg")
import pywt
import csv


def save_data_to_csv(data_dict, filename):
    # 提取字段名
    fields = ['distance', 'entropy', 'lap', 'var', 'cosine', 'rcf']

    # 确保所有数据长度一致
    assert all(len(data_dict[key]) == len(data_dict['distance']) for key in fields)

    # 写入CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入标题行
        writer.writerow(fields)
        # 逐行写入数据（处理NaN为空白）
        for i in range(len(data_dict['distance'])):
            row = [
                data_dict['distance'][i],
                data_dict['entropy'][i],
                data_dict['lap'][i],
                data_dict['var'][i],
                data_dict['cosine'][i] if not np.isnan(data_dict['cosine'][i]) else '',
                data_dict['rcf'][i]
            ]
            writer.writerow(row)


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
        image_array = image_array[:h-1, :]
    if w % 2 != 0:
        image_array = image_array[:, :w-1]
    # 进行一层小波分解（使用haar小波）
    coeffs = pywt.dwt2(image_array, 'haar')
    cA, (cH, cV, cD) = coeffs
    # 计算高频细节系数的能量（平方和）
    energy = (cH**2 + cV**2 + cD**2).sum()
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


def compute_all_criterions(image_folder):
    # 获取所有图像文件并解析距离值
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 解析距离值并过滤无效文件
    valid_files = []
    for f in image_files:
        distance = parse_distance(f)
        if distance is not None:
            valid_files.append((f, distance))

    # 按距离值升序排序（确保处理顺序正确）
    valid_files.sort(key=lambda x: x[1])
    sorted_files = [f[0] for f in valid_files]
    sorted_distances = [f[1] for f in valid_files]

    data_points = []
    prev_image = None  # 用于保存前一张图像（按距离顺序）

    for idx, (image_file, distance) in enumerate(zip(sorted_files, sorted_distances)):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("L")

        # 统一调整图像尺寸（避免后续resize影响CS计算）
        # image = image.resize((256, 256), Image.BICUBIC)  # 固定尺寸+高质量插值
        image_array = np.array(image, dtype=np.float32)

        # 计算指标（熵、LAP、VAR）
        entropy = calculate_entropy(image_array)
        lap = calculate_lap(image_array)
        var = calculate_var(image_array)

        # 新增小波计算
        wavelet = calculate_wavelet(image_array)

        # 计算RCF
        rcf_v = rcf_value(rcf_model, image_array, 'F:\dongjiayao\Pycharm\Holo-Track\img\edge1')

        # 计算余弦相似度（仅当非第一张图像时）
        cosine_sim = np.nan
        if prev_image is not None:
            # 直接使用统一尺寸后的图像数组
            prev_array = np.array(prev_image)
            current_array = image_array
            cosine_sim = calculate_cosine_similarity(prev_array, current_array)

        data_points.append({
            'distance': distance,
            'entropy': entropy,
            'lap': lap,
            'var': var,
            'cosine': cosine_sim,
            'wavelet': wavelet,
            'rcf': rcf_v,
            'filename': image_file  # 用于调试
        })
        prev_image = image  # 保存当前图像供下一帧使用

    # 验证排序结果
    print("前5个距离值:", [p['distance'] for p in data_points[:5]])

    return {
        'distance': [p['distance'] for p in data_points],
        'entropy': [p['entropy'] for p in data_points],
        'lap': [p['lap'] for p in data_points],
        'var': [p['var'] for p in data_points],
        'cosine': [p['cosine'] for p in data_points],
        'rcf': [p['rcf'] for p in data_points],
        'wavelet': [p['wavelet'] for p in data_points]
    }


# 归一化数据
def normalize_data(data_dict):
    normalized_data = {}
    for key, values in data_dict.items():
        if key == 'distance':
            normalized_data[key] = values
        # elif key == 'cosine':
        #     values = [x * 1e10 for x in values]
        #
        #     min_val = np.nanmin(values)
        #     max_val = np.max(values)
        #     normalized = (values - min_val) / (max_val - min_val + 1e-10)  # 归一化
        #     normalized_data[key] = normalized
        else:
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            normalized = (values - min_val) / (max_val - min_val + 1e-10)  # 归一化
            normalized_data[key] = normalized
    return normalized_data


# 绘制曲线图
def plot_criterions(data_dict):
    # 设置全局字体大小
    plt.rcParams['font.size'] = 14  # 调整字体大小为14
    plt.rcParams['xtick.labelsize'] = 14  # X轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 14  # Y轴刻度字体大小
    plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
    plt.rcParams['legend.fontsize'] = 14  # 图例字体大小

    plt.figure(figsize=(10, 6))
    plt.xlabel('Distance')
    plt.ylabel('Normalized Value')
    plt.title('Comparison of Different Criteria')

    # 绘制熵曲线
    # entropy_line, = plt.plot(data_dict['distance'], data_dict['entropy'], label='Entropy', marker='o', linestyle='-',
    #                          markersize=1)

    # 绘制拉普拉斯曲线
    # lap_line, = plt.plot(data_dict['distance'], data_dict['lap'], label='LAP', marker='s', linestyle='--', markersize=1)

    # 绘制方差曲线
    # var_line, = plt.plot(data_dict['distance'], data_dict['var'], label='VAR', marker='^', linestyle='-.', markersize=1)

    # 绘制余弦相似度曲线
    # cosine_line, = plt.plot(data_dict['distance'], data_dict['cosine'], label='Cosine Similarity', marker='x',
    #                         linestyle=':', markersize=1)

    # 绘制RCF曲线
    rcf_line, = plt.plot(data_dict['distance'], data_dict['rcf'], label='RCF', marker='o', linestyle='-', markersize=1)

    # 绘制小波曲线
    # wavelet_line, = plt.plot(data_dict['distance'], data_dict['wavelet'], label='Wavelet', marker='x', linestyle='--', markersize=1)

    plt.legend()
    plt.grid(True)

    try:
        import mplcursors
        mplcursors.cursor(hover=True)
    except ImportError:
        print("mplcursors not installed. Install it with 'pip install mplcursors' for enhanced features.")

    plt.show()


if __name__ == "__main__":
    rcf_model = RCF()
    rcf_model.cuda()
    rcf_model.eval()

    checkpoint = torch.load(r'F:\dongjiayao\Pycharm\Holo-Track\rcf\trained_model\checkpoint_epoch29.pth', weights_only=True)
    rcf_model.load_state_dict(checkpoint['state_dict'])

    # 示例：读取文件夹中的所有图像并计算指标
    # image_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\2\cut2"  # 实际颗粒
    # image_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\1\cut1"  # 实际颗粒
    # image_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\3\tmp"  # 高斯模糊模拟
    # image_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\3\tmp\tmp1"  # angular模拟
    image_folder = r"F:\dongjiayao\Data\VOC\angular\2007_000032"  # VOC
    data = compute_all_criterions(image_folder)
    # 保存原始数据到CSV
    # save_data_to_csv(data, 'output_raw.csv')

    normalized_data = normalize_data(data)
    # save_data_to_csv(normalized_data, 'output_norm.csv')

    plot_criterions(normalized_data)
