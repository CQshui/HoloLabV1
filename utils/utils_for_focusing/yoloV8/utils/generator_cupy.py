import os
import cupy as cp  # 使用 CuPy 替代 NumPy
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm  # 导入 tqdm 库

# 参数设置
wavelength = 532e-9  # 波长，单位：米
pixel_size = 0.098e-6  # 像素尺寸，单位：米
num_reconstructions = 50  # 每张全息图重建的图像数量
reconstruction_range = 0.0003  # 重建范围，单位：米

# 主文件夹路径
main_folder = r"E:\DongJiayao\Data\AutoFocusExperiment\deepsort_train_data\originV1"
holograms_folder = os.path.join(main_folder, "holograms")
reconstruction_folder = os.path.join(main_folder, "reconstruction")
os.makedirs(reconstruction_folder, exist_ok=True)

# 初始化CSV文件
csv_path = os.path.join(main_folder, "metadata.csv")
if os.path.exists(csv_path):
    # 如果 CSV 文件已存在，读取已有数据
    df = pd.read_csv(csv_path)
else:
    # 如果 CSV 文件不存在，创建新的 DataFrame
    df = pd.DataFrame(columns=["ID", "Hologram", "Focused", "Distance"])

# 全局变量，用于手动截取频谱
point1, point2 = None, None
U0 = None
cut_size = []  # 用于保存频谱截取位置


def on_mouse(event, x, y, flags, param):
    global point1, point2, U0, cut_size, delta_x, delta_y
    img2 = cp.asnumpy(cp.log(cp.abs(U0) + 1))  # 显示频谱的对数幅度
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 3)
        cv2.imshow('spectrum', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 3)
        cv2.imshow('spectrum', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 8)
        cv2.imshow('spectrum', img2)

        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        rectangle_width = abs(point1[0] - point2[0])
        rectangle_height = abs(point1[1] - point2[1])

        # 记录截取区域的坐标
        cut_size = [min_x, min_y, rectangle_width, rectangle_height]

        # 截取频谱
        U0[0:min_y, min_x:img_width] = 0
        U0[min_y:img_height, min_x + rectangle_width:img_width] = 0
        U0[min_y + rectangle_height:img_height, 0:min_x + rectangle_width] = 0
        U0[0:min_y + rectangle_height, 0:min_x] = 0

        img2[0:min_y, min_x:img_width] = 0
        img2[min_y:img_height, min_x + rectangle_width:img_width] = 0
        img2[min_y + rectangle_height:img_height, 0:min_x + rectangle_width] = 0
        img2[0:min_y + rectangle_height, 0:min_x] = 0

        # 显示截取后的频谱
        cv2.imshow('spectrum', img2)

        # 滤出最中心的高亮像素块
        _, binary_image = cv2.threshold(img2, 180, 255, cv2.THRESH_BINARY)
        binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
        contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter_x = []
        counter_y = []
        counter_w = []
        counter_h = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            counter_x.append(x)
            counter_y.append(y)
            counter_w.append(w)
            counter_h.append(h)
        # 找到最大宽度的矩形并画出
        max_index = counter_w.index(max(counter_w))
        x_center = counter_x[max_index]
        y_center = counter_y[max_index]
        w_center = counter_w[max_index]
        h_center = counter_h[max_index]
        cv2.rectangle(img2, (x_center, y_center), (x_center + w_center, y_center + h_center),
                      (0, 255, 0), 8)

        cv2.namedWindow('spectrum', 0)
        cv2.imshow('spectrum', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        delta_x = int(0.5 * w_center + x_center - 0.5 * img_width)
        delta_y = int(0.5 * h_center + y_center - 0.5 * img_height)
        cut_size.extend([delta_x, delta_y])


def auto_adjust():
    global U0
    # 滤出最中心的高亮像素块
    img2 = cp.asnumpy(cp.log(cp.abs(U0) + 1))  # 显示频谱的对数幅度
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img2[0:min_y, min_x:img_width] = 0
    img2[min_y:img_height, min_x + rectangle_width:img_width] = 0
    img2[min_y + rectangle_height:img_height, 0:min_x + rectangle_width] = 0
    img2[0:min_y + rectangle_height, 0:min_x] = 0

    _, binary_image = cv2.threshold(img2, 180, 255, cv2.THRESH_BINARY)
    binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
    contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter_x = []
    counter_y = []
    counter_w = []
    counter_h = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        counter_x.append(x)
        counter_y.append(y)
        counter_w.append(w)
        counter_h.append(h)
    # 找到最大宽度的矩形并画出
    max_index = counter_w.index(max(counter_w))
    x_center = counter_x[max_index]
    y_center = counter_y[max_index]
    w_center = counter_w[max_index]
    h_center = counter_h[max_index]
    cv2.rectangle(img2, (x_center, y_center), (x_center + w_center, y_center + h_center),
                  (0, 255, 0), 8)

    if len(cut_size) == 0:
        cv2.namedWindow('spectrum', 0)
        cv2.imshow('spectrum', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    delta_x = int(0.5 * w_center + x_center - 0.5 * img_width)
    delta_y = int(0.5 * h_center + y_center - 0.5 * img_height)

    U0[0:min_y, min_x:img_width] = 0
    U0[min_y:img_height, min_x + rectangle_width:img_width] = 0
    U0[min_y + rectangle_height:img_height, 0:min_x + rectangle_width] = 0
    U0[0:min_y + rectangle_height, 0:min_x] = 0
    U0 = cp.roll(U0, -delta_x, axis=1)
    U0 = cp.roll(U0, -delta_y, axis=0)
    # cut_size.extend([delta_x, delta_y])


def cut(img):
    global U0, cut_size, delta_x, delta_y
    img_gpu = cp.asarray(img)  # 将 NumPy 数组转换为 CuPy 数组
    U0 = cp.fft.fftshift(cp.fft.fft2(img_gpu))  # 使用 CuPy 的 FFT 函数
    cv2.namedWindow('spectrum', 0)
    cv2.setMouseCallback('spectrum', on_mouse)
    img2 = cp.asnumpy(cp.log(cp.abs(U0) + 1))  # 显示频谱的对数幅度
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('spectrum', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 移动频谱到中心
    # if len(cut_size) > 0:
    min_x, min_y, rectangle_width, rectangle_height, delta_x, delta_y = cut_size
    delta_x = int(0.5 * rectangle_width + min_x - 0.5 * img_width)
    delta_y = int(0.5 * rectangle_height + min_y - 0.5 * img_height)
    U0 = cp.roll(U0, -delta_x, axis=1)
    U0 = cp.roll(U0, -delta_y, axis=0)

    return U0

# 遍历holograms文件夹
subfolders = [f for f in os.listdir(holograms_folder) if os.path.isdir(os.path.join(holograms_folder, f))]

# 统计所有需要重建的图像数量
total_images = 0
for subfolder in subfolders:
    hologram_path = os.path.join(holograms_folder, subfolder)
    total_holograms = [f for f in os.listdir(hologram_path)]
    total_images += len(total_holograms) * num_reconstructions

# 使用 tqdm 显示进度条
with tqdm(total=total_images, desc="重建进度") as pbar:
    for idx, subfolder in enumerate(subfolders):
        # 获取全息图聚焦位置
        focus_position = float(subfolder)  # 子文件夹名称为聚焦位置
        hologram_path = os.path.join(holograms_folder, subfolder)

        total_holograms = [f for f in os.listdir(hologram_path)]
        for idx, hologram_name in enumerate(total_holograms):

            hologram = cv2.imread(os.path.join(hologram_path, hologram_name), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = hologram.shape[:2]
            # 第一次处理时手动截取频谱
            if len(cut_size) == 0:
                U0 = cut(hologram)  # 手动截取频谱

            # 后续处理时根据截取位置截取频谱
            else:
                hologram_gpu = cp.asarray(hologram)  # 将 NumPy 数组转换为 CuPy 数组
                U0 = cp.fft.fftshift(cp.fft.fft2(hologram_gpu))  # 使用 CuPy 的 FFT 函数
                if len(cut_size) > 0:
                    min_x, min_y, rectangle_width, rectangle_height, delta_x, delta_y = cut_size
                    # U0[0:min_y, min_x:img_width] = 0
                    # U0[min_y:img_height, min_x + rectangle_width:img_width] = 0
                    # U0[min_y + rectangle_height:img_height, 0:min_x + rectangle_width] = 0
                    # U0[0:min_y + rectangle_height, 0:min_x] = 0
                    # U0 = cp.roll(U0, -delta_x, axis=1)
                    # U0 = cp.roll(U0, -delta_y, axis=0)
                    auto_adjust()

            # 创建重建文件夹
            reconstruction_subfolder = os.path.join(reconstruction_folder, hologram_name)
            os.makedirs(reconstruction_subfolder, exist_ok=True)

            # 生成重建位置
            focus_position_gpu = cp.asarray([focus_position])  # 将聚焦位置转换为 CuPy 数组
            random_offsets = cp.random.uniform(-reconstruction_range, reconstruction_range, num_reconstructions - 1)
            reconstruction_positions = cp.sort(
                cp.concatenate([
                    focus_position_gpu,  # 聚焦位置
                    focus_position_gpu + random_offsets  # 随机偏移位置
                ])
            )

            # 角谱重建
            for i, z in enumerate(reconstruction_positions):
                # 计算传播相位
                k = 2 * cp.pi / wavelength
                fx = cp.fft.fftfreq(hologram.shape[1], d=pixel_size)
                fy = cp.fft.fftfreq(hologram.shape[0], d=pixel_size)
                FX, FY = cp.meshgrid(fx, fy)

                temp = cp.clip(1 - ((wavelength * FX) ** 2 + (wavelength * FY) ** 2), 0, None)

                # 重建图像
                g = cp.exp(1j * k * z * cp.sqrt(temp))
                g[temp < 0] = 0
                g = cp.fft.fftshift(g)

                U0_p = cp.fft.ifft2(cp.fft.ifftshift(U0))
                U0_p = cp.fft.fftshift(cp.fft.fft2(U0_p))
                U0_p = U0_p * g
                U0_p = cp.fft.ifft2(cp.fft.ifftshift(U0_p))
                reconstructed_image = cp.abs(U0_p)

                # 保存重建图像
                image_name = f"{i + 1:02d}_z_{z:.6f}m.png"
                image_path = os.path.join(reconstruction_subfolder, image_name)
                result = cv2.normalize(cp.asnumpy(reconstructed_image / cp.max(reconstructed_image)),
                                       None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(image_path, result)

                # 记录聚焦重建图名称
                if z == focus_position:
                    new_row = pd.DataFrame({
                        "ID": [idx + 1],
                        "Hologram": [hologram_name],
                        "Focused": [image_name],
                        "Distance": [subfolder]  # 保存子文件夹名称（聚焦距离）
                    })
                    # 使用 pd.concat 追加数据
                    df = pd.concat([df, new_row], ignore_index=True)
                    # 保存到 CSV 文件
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

                # 更新进度条
                pbar.update(1)

print("数据集准备完成！")
