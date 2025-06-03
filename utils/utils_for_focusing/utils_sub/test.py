import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cosine


def cosine_score(image1, image2):
    """
    计算两幅图像的余弦分值（Cosine Score）
    :param image1: 第一幅图像（NumPy数组）
    :param image2: 第二幅图像（NumPy数组）
    :return: 余弦分值
    """
    # 确保图像为灰度图
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 将图像展平为一维向量
    vector1 = image1.flatten().astype(np.float32)
    vector2 = image2.flatten().astype(np.float32)

    # 计算点积（分子）
    dot_product = np.dot(vector1, vector2)

    # 计算向量的模（分母）
    norm1 = np.sqrt(np.sum(vector1**2))
    norm2 = np.sqrt(np.sum(vector2**2))

    # 计算余弦相似度
    if norm1 == 0 or norm2 == 0:
        return 0  # 避免除以零
    cosine_sim = dot_product / (norm1 * norm2)
    return cosine_sim


def grad_focus_score(image):
    """
    计算灰度梯度聚焦分数
    :param image: 输入图像（灰度图）
    :return: 灰度梯度聚焦分数
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # 计算平均梯度幅值
    focus_score = np.sum(grad_magnitude) / (image.shape[0] * image.shape[1])
    return focus_score


def calculate_cosine_scores(image_stack):
    """
    计算图像堆栈中相邻图像的余弦分值
    :param image_stack: 图像堆栈（列表或 NumPy 数组，形状为 (N, H, W)）
    :return: 余弦分值列表
    """
    cosine_scores = []
    for i in range(len(image_stack) - 1):
        score = cosine_score(image_stack[i], image_stack[i + 1])
        cosine_scores.append(score)
    return cosine_scores


def calculate_grad_scores(image_stack):
    """
    计算图像堆栈中每幅图像的灰度梯度聚焦分数
    :param image_stack: 图像堆栈（列表或 NumPy 数组，形状为 (N, H, W)）
    :return: 灰度梯度聚焦分数列表
    """
    grad_scores = []
    for image in image_stack:
        score = grad_focus_score(image)
        grad_scores.append(score)
    return grad_scores


def extract_number(filename):
    """
    从文件名中提取数字部分
    :param filename: 文件名（如 offaxis_99_-0.0000249.png 或 1.png）
    :return: 数字（如 99 或 1）
    """
    # 尝试匹配 offaxis_99_-0.0000249.png 格式
    match = re.search(r'offaxis_(\d+)_', filename)
    if match:
        return int(match.group(1))
    # 尝试匹配 1.png 格式
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0  # 如果没有匹配到数字，返回0


def calculate_focus_with_folder(folder_path, method='cs'):
    """
    计算文件夹中所有图像的聚焦分数
    :param folder_path: 文件夹路径
    :param method: 聚焦评估方法，'cs' 表示余弦分值，'grad' 表示灰度梯度
    :return: 字典，键为图像文件名，值为聚焦分数
    """
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        raise ValueError("文件夹中没有图像文件")

    # 按照文件名中的数字排序
    image_files.sort(key=extract_number)

    # 读取所有图像
    image_stack = []
    for image_file in tqdm(image_files, desc="Loading Images", unit="image"):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"警告：无法读取图像 {image_file}，跳过")
            continue
        image_stack.append(image)

    # 根据方法计算聚焦分数
    if method == 'cs':
        focus_scores = calculate_cosine_scores(image_stack)
    elif method == 'grad':
        focus_scores = calculate_grad_scores(image_stack)
    else:
        raise ValueError("未知的聚焦评估方法，请选择 'cs' 或 'grad'")

    # 将聚焦分数与图像文件名对应
    focus_results = {}
    for i in range(len(focus_scores)):
        focus_results[image_files[i]] = focus_scores[i]

    return focus_results


def plot_focus_scores(focus_results, method='cs'):
    """
    绘制聚焦分数曲线图
    :param focus_results: 聚焦分数结果字典，键为图像文件名，值为聚焦分数
    :param method: 聚焦评估方法，'cs' 表示余弦分值，'grad' 表示灰度梯度
    """
    # 提取文件名和聚焦分数
    filenames = list(focus_results.keys())
    focus_scores = list(focus_results.values())
    # 提取数字作为横坐标
    indices = [extract_number(f) for f in filenames]
    # 绘制曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(indices, focus_scores, marker='o', linestyle='-', color='b')
    plt.title(f"Focus Score Curve ({'Cosine Score' if method == 'cs' else 'Gradient'} Method)")
    plt.xlabel("Image Index")
    plt.ylabel(f"Focus Score ({'Cosine Score' if method == 'cs' else 'Gradient'})")
    plt.grid(True)
    plt.show()


def calculate_statistics(focus_scores):
    """
    计算聚焦分数的统计信息
    :param focus_scores: 聚焦分数列表
    :return: 最大范围, 95% 置信区间
    """
    # 计算最大范围
    max_range = (np.min(focus_scores), np.max(focus_scores))
    # 计算95%置信区间
    mean = np.mean(focus_scores)
    std_err = stats.sem(focus_scores)  # 标准误差
    confidence_interval = stats.t.interval(0.95, len(focus_scores) - 1, loc=mean, scale=std_err)
    return max_range, confidence_interval


def find_outliers(focus_results, confidence_interval):
    """
    找出聚焦分数在置信区间外的图像
    :param focus_results: 聚焦分数结果字典，键为图像文件名，值为聚焦分数
    :param confidence_interval: 95% 置信区间
    :return: 置信区间外的图像名称和分数列表
    """
    outliers = []
    lower_bound, upper_bound = confidence_interval
    for image_file, focus_value in focus_results.items():
        if focus_value < lower_bound or focus_value > upper_bound:
            outliers.append((image_file, focus_value))
    return outliers


if __name__ == '__main__':
    # 文件夹路径
    folder_path = r'F:\dongjiayao\Pycharm\Holo-Track\img\cut1'

    # 用户选择聚焦评估方法
    method = input("请选择聚焦评估方法（输入 'cs' 或 'grad'）：").strip().lower()
    if method not in ['cs', 'grad']:
        raise ValueError("未知的聚焦评估方法，请选择 'cs' 或 'grad'")

    # 计算聚焦分数
    focus_results = calculate_focus_with_folder(folder_path, method=method)

    # 输出结果
    for image_file, focus_value in focus_results.items():
        print(f"{image_file} 的聚焦分数为: {focus_value:.4f}")

    # 绘制聚焦分数曲线图
    plot_focus_scores(focus_results, method=method)

    # 计算统计信息
    focus_scores = list(focus_results.values())
    max_range, confidence_interval = calculate_statistics(focus_scores)
    print(f"聚焦分数的最大范围: {max_range[0]:.4f} 到 {max_range[1]:.4f}")
    print(f"聚焦分数的95%置信区间: {confidence_interval[0]:.4f} 到 {confidence_interval[1]:.4f}")

    # 找出置信区间外的图像
    outliers = find_outliers(focus_results, confidence_interval)
    if outliers:
        print("\n聚焦分数在95%置信区间外的图像：")
        for image_file, focus_value in outliers:
            print(f"{image_file} 的聚焦分数为: {focus_value:.4f}")
    else:
        print("\n所有图像的聚焦分数均在95%置信区间内。")

