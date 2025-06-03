"""
用于给VOC数据集创建全息图，用rcf_predict_batch.py测试并获取聚焦曲线随着z的变化趋势图，进一步验证聚焦判据的可行性
"""
import os
import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2


def angular_spectrum_propagation(input_amp, wavelength, pixel_size, distance):
    """
    使用角谱方法计算光场的传播
    :param input_amp: 输入振幅图像（归一化到0-1）
    :param wavelength: 光波长（单位：米）
    :param pixel_size: 像素尺寸（单位：米）
    :param distance: 传播距离（单位：米）
    :return: 输出振幅图像
    """
    ny, nx = input_amp.shape
    x = np.arange(nx) * pixel_size
    y = np.arange(ny) * pixel_size
    fx = np.fft.fftfreq(nx, d=pixel_size)
    fy = np.fft.fftfreq(ny, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    # 计算传递函数，避免倏逝波
    H = np.exp(1j * k * distance * np.sqrt(1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)))
    H[np.sqrt(FX ** 2 + FY ** 2) >= 1 / wavelength] = 0  # 去除倏逝波

    input_field = input_amp.astype(np.complex64)
    fft_field = fft2(input_field)
    output_field = ifft2(fft_field * H)
    output_amp = np.abs(output_field)
    return output_amp


# 参数配置
input_folder = r'F:\dongjiayao\Data\VOC\SegmentationImage'
output_base = r'F:\dongjiayao\Data\VOC\angular'
wavelength = 0.5e-6  # 500nm
pixel_size = 3.45e-6  # 3.45微米
distances = np.linspace(0, 0.003, 50)  # 生成50个均匀距离点

os.makedirs(output_base, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0

            base_name = os.path.splitext(filename)[0]
            output_folder = os.path.join(output_base, base_name)
            os.makedirs(output_folder, exist_ok=True)

            index = 0
            for z in distances:
                propagated_amp = angular_spectrum_propagation(
                    img_array, wavelength, pixel_size, z
                )
                output_img = (propagated_amp / propagated_amp.max() * 255).astype(np.uint8)

                # 修改文件名生成方式，保留3位小数精度
                z_mm = z * 1000  # 转换为毫米
                output_filename = f"off_{index:d}_{z_mm:.3f}.jpg"  # 例：1.500mm
                output_path = os.path.join(output_folder, output_filename)
                Image.fromarray(output_img).save(output_path)
                index += 1
            print(f"处理完成：{filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错：{e}")