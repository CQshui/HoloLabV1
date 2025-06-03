from module.hologram import Hologram
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import os
import cv2
import numpy as np
import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MockData:
    def __init__(self, data_path=None):
        ''''''
        '''直接构造变量：从某个文件夹中获取'''
        self.data_path = data_path if data_path else r'D:\Development\HoloLab\test_data'
        self.method = "polar"
        '''定义一个hologram实例，并直接给定参数，用于测试'''
        self.Define_Hologram()
        # self.Get_Raw_Hologram_Image()
        # self.Get_Reconstructed_Images()
        # self.Get_Focusing_Image()
        # self.Make_Wavefront_and_Phase()
        # self.Get_Focusing_Single_Image()
        self.Get_Segmented_Particles()

    def run(self, config):
        class_name = self.__class__.__name__
        print(f"Running operation in class: {class_name}")

    def Define_Hologram(self):
        '''预定义一个hologram实例'''
        key_config = {
            'image_path': None,
            'image_name': None,
            'hologram_type': 'inline',  # inline, offaxis
            'pixel_size': unit_um * 0.46,
            'wavelength': unit_nm * 532,
            'z_start': unit_mm * -1.2,
            'z_end': unit_mm * 0.8,
            'z_step': unit_mm * 0.5,
            'density': 2.7,
            'method': 'polar'   # onlyholo, polar
        }
        self.holo = Hologram(key_config)

    def Get_Raw_Hologram_Image(self):
        sub_folder = 'hologram'
        file_name = 'inline_single_particle_Z0.030.jpg'
        file_path = os.path.join(self.data_path, sub_folder, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        self.holo.hologram_raw = image
        self.holo.hologram_type = 'inline'  # inline, offaxis
        self.holo.image_path = os.path.join(self.data_path, sub_folder)
        self.holo.image_name = file_name

        self.holo.pixel_num_x = self.holo.hologram_raw.shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.hologram_raw.shape[0]  # 图片高height

    def Get_PreProcessing_Image(self):
        sub_folder = 'preprocessing'
        file_name = 'inline_single_particle_Z0.030.jpg'
        file_path = os.path.join(self.data_path, sub_folder, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        self.holo.hologram_pro = image

        self.holo.pixel_num_x = self.holo.hologram_raw.shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.hologram_raw.shape[0]  # 图片高height

    def Get_Reconstructed_Images(self):
        sub_folder = 'reconstruction'
        folder_path = os.path.join(self.data_path, sub_folder)

        # 获取所有图像文件，按名称排序
        file_list = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        if not file_list:
            raise FileNotFoundError(f"No image files found in {folder_path}")

        z_list, image_list = [], []
        for file_name in file_list:

            '从 plane_0.000100.jpg 中提取 z=0.000100（单位 m）'
            # 名字必须是这样的形式：plane_0.000100.jpg
            z_str = file_name.split('_')[1].split('.')[0] + '.' + file_name.split('_')[1].split('.')[1]
            z_val = float(z_str)
            z_list.append(z_val)

            '读取图像'
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            image_list.append(image)

        # 保存图像信息
        # 当你用opencv读取图片时，得到的 numpy 数组的 .shape 是 (H, W)
        self.holo.reconstruction = np.stack(image_list, axis=-1)  # shape: (H, W, N) | axis=0，则 shape: (N, H, W)

        # 保存 z 信息
        z_array = np.array(z_list)
        self.holo.z_array = z_array
        self.holo.z_num = len(z_array)
        self.holo.z_start = z_array[0]
        self.holo.z_end = z_array[-1]

        self.holo.pixel_num_x = self.holo.reconstruction[:, :, 0].shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.reconstruction[:, :, 0].shape[0]  # 图片高height

    def Get_Focusing_Image(self):
        sub_folder = 'focusing'
        file_name = 'inline_single_particle_Z0.030.jpg'
        file_path = os.path.join(self.data_path, sub_folder, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        self.holo.focusing = image

        self.holo.pixel_num_x = self.holo.hologram_raw.shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.hologram_raw.shape[0]  # 图片高height

    def Make_Wavefront_and_Phase(self):
        def zernike_radial(n, m, rho):
            R = np.zeros_like(rho)
            for k in range((n - abs(m)) // 2 + 1):
                c = (-1) ** k * math.factorial(n - k)
                c /= (math.factorial(k) *
                      math.factorial((n + abs(m)) // 2 - k) *
                      math.factorial((n - abs(m)) // 2 - k))
                R += c * rho ** (n - 2 * k)
            return R

        def zernike(n, m, rho, theta):
            R = zernike_radial(n, m, rho)
            if m >= 0:
                return R * np.cos(m * theta)
            else:
                return R * np.sin(-m * theta)

        '图像坐标'
        H, W = 512, 512
        img_x = np.linspace(-1, 1, W)
        img_y = np.linspace(-1, 1, H)
        mesh_x, mesh_y = np.meshgrid(img_x, img_y)
        rho = np.sqrt(mesh_x ** 2 + mesh_y ** 2)
        theta = np.arctan2(mesh_y, mesh_x)

        '设置超出单位圆的区域为0'
        mask = rho <= 1.0

        '生成泽尼克相位（比如 n=4, m=2，像散项）'
        phase = np.zeros_like(rho)
        phase[mask] = zernike(n=4, m=2, rho=rho[mask], theta=theta[mask])
        phase = phase * 5  # 人为再变大

        '振幅为1，相位为泽尼克'
        wavefront = np.exp(1j * phase)
        self.holo.wavefront = wavefront
        self.holo.wave_front_ini = wavefront
        self.holo.phase = np.angle(wavefront)

        '更新图像尺寸参数'
        self.holo.pixel_num_x = self.holo.phase.shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.phase.shape[0]  # 图片高height

        # plt.imshow(np.angle(wavefront), cmap='jet')
        # plt.colorbar()
        # plt.title("Wavefront Phase")
        # plt.show()

    def Get_Focusing_Single_Image(self):
        """
        为 Identification 模块提供批量分类的图像路径
        返回一个列表，列表元素是 (subdir, image_path)
        """
        sub_folder = 'focus_single'
        folder = os.path.join(self.data_path, sub_folder)
        image_list = []
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Identification folder not found: {folder}")

        if self.holo.method == "holo":

            files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ])
            for f in files:
                image_list.append((sub_folder, os.path.join(folder, f)))

        elif self.holo.method == "polar":

            # 以dolp目录为基准获取文件列表
            dolp_folder = os.path.join(folder, 'dolp')
            if not os.path.isdir(dolp_folder):
                raise FileNotFoundError(f"Polarization dolp folder not found: {dolp_folder}")

            files = sorted([
                f for f in os.listdir(dolp_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ])

            for f in files:
                # 检查其他两个通道是否存在
                amp_path = os.path.join(folder, 'amplitude', f)
                aop_path = os.path.join(folder, 'aop', f)
                if not (os.path.exists(amp_path) and os.path.exists(aop_path)):
                    print(f"Skip {f}: missing amplitude or aop image")
                    continue

                image_list.append((sub_folder, os.path.join(dolp_folder, f)))

        # 保存到 Hologram 对象，后续可用
        self.holo.focusing_single = image_list
        self.holo.pixel_num_x = self.holo.hologram_raw.shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.hologram_raw.shape[0]  # 图片高height

    def Get_Segmented_Particles(self):
        """加载分割后的颗粒图像"""
        sub_folder = 'segmentation_each'
        folder_path = os.path.join(self.data_path, sub_folder)
        image_list = []

        if not os.path.exists(folder_path):
            print(f"  └─Segmentation folder not found: {folder_path}")
            return

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        '读取图像'
        for f in files:
            file_path = os.path.join(folder_path, f)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            image_list.append(image)

        self.holo.segmentation_each = image_list
        self.holo.pixel_num_x = self.holo.hologram_raw.shape[1]  # 图片宽width
        self.holo.pixel_num_y = self.holo.hologram_raw.shape[0]  # 图片高height


if __name__ == '__main__':
    hologram = MockData()
    print(f'- Hologram Data loaded successfully: {hologram.data_path}')
    print(f'  ├─Hologram Raw  Image')
    print(f'  ├─PreProcessing Image')
    print(f'  ├─Reconstructed Image')
    print(f'  ├─Focusing      Image')
    print(f'  ├─Wavefront and Phase')
