from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import cv2
import numpy as np
from numpy.fft import fftshift, fft2, ifft2, ifftshift

import matplotlib
import matplotlib.pyplot as plt

try:
    import torch
except Exception as e:
    print(f"[Module] {e}")

class Reconstruction():
    def __init__(self, hologram, config):
        # self.hologram_raw   =
        self.Holo_class     = hologram  # 整个hologram类
        self.hologram       = hologram.hologram
        self.spectrum       = hologram.hologram_spectrum

        self.pixel_size     = config.image_info['pixel_size']   * unit_um
        self.wavelength     = config.image_info['wavelength']   * unit_nm

        self.method         = config.reconstruction['method']
        self.z_start        = config.reconstruction['z_start']  * unit_mm
        self.z_end          = config.reconstruction['z_end']    * unit_mm
        self.z_step         = config.reconstruction['z_step']   * unit_mm
        self.z_num          = int((self.z_end - self.z_start) / self.z_step) + 1
        self.z_array        = np.linspace(self.z_start, self.z_end, self.z_num)
        self.z_list         = self.z_array.tolist()
        self.filter_method  = config.spectrum['method']

        self.cpu_num        = config.reconstruction['cpu_num']     # TODO 多线程
        self.gpu_num        = config.reconstruction['gpu_num']     # TODO 多GPU

        self.type           = config.file_info['holo_type']

        # 每一个截面的图像，保存成list
        self.reconstruction_list = []

        '''最后返回给前端，每一个截面的图像，为了便于前端展示，弄成字典形式，key为z，value为图像'''
        self.reconstruction_dict = {}

    def run(self):
        if self.method == 'Angular_CPU':
            self.Reconstruction_Angular_Spectrum_CPU()
        elif self.method == 'Angular_GPU':
            self.Reconstruction_Angular_Spectrum_GPU()
        else:
            self.Reconstruction_Example()

        return self._make_reconstruction_dict()

    def _make_reconstruction_dict(self):
        # 判断z的量级
        unit = 1.0
        if self.z_step >= 1e-4:
            unit = unit_mm
            unit_str = 'mm'
        else:
            unit = unit_um
            unit_str = 'um'

        for z, image in zip(self.z_list, self.reconstruction_list):
            key_z = f"Z {z / unit:.3f} {unit_str}"
            self.reconstruction_dict[key_z] = image

            # plt.imsave(r'C:\Users\lzmfor\Desktop\AAAAA/plane_{}.jpg'.format(z), np.abs(image), cmap="gray")  # 重建后图像保存路径

        return self.reconstruction_dict

    def Reconstruction_Example(self):
        for z in self.z_array:
            image = np.random.rand(20, 20)
            self.reconstruction_list.append(image)

    def Reconstruction_Angular_Spectrum_CPU(self):
        if self.type == 'Inline':
            print('同轴处理')
            self.current = ifft2(ifftshift(self.spectrum))

        else:
            print('离轴处理')
            self.current = ifft2(ifftshift(self.spectrum))

        # 图像尺寸
        width, height = self.current.shape[1], self.current.shape[0]

        # 空间频率
        fx = np.linspace(-1 / (2 * self.pixel_size), 1 / (2 * self.pixel_size), width)
        fy = np.linspace(-1 / (2 * self.pixel_size), 1 / (2 * self.pixel_size), height)
        FX, FY = np.meshgrid(fx, fy)
        temp = 1 - ((self.wavelength * FX) ** 2 + (self.wavelength * FY) ** 2)
        temp[temp < 0] = 0

        z = np.linspace(self.z_start, self.z_end,
                        int((self.z_end - self.z_start) / self.z_step) + 1)

        # 开始重建
        for i in range(len(z)):
            g = np.exp(1j * (2 * np.pi / self.wavelength) * z[i] * np.sqrt(temp))
            g[temp < 0] = 0
            g = fftshift(g)
            u = fft2(fftshift(self.current))
            u = u * g
            u = ifftshift(ifft2(u))

            self.reconstruction_list.append(u)
            # plt.imsave(r'E:\Projects\HoloLab\test_data\tmp/{:02d}plane_{:.3f}.jpg'.format(i, z[i]), abs(u), cmap="gray")  # 重建后图像保存路径


    def Reconstruction_Angular_Spectrum_GPU(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.type == 'Inline':
            print('同轴处理')
            self.spectrum = torch.tensor(self.spectrum, device=device, dtype=torch.complex64)
            self.current = torch.fft.ifft2(torch.fft.ifftshift(self.spectrum))
        else:
            print('离轴处理')
            self.spectrum = torch.tensor(self.spectrum, device=device, dtype=torch.complex64)
            self.current = torch.fft.ifft2(torch.fft.ifftshift(self.spectrum))

        height, width = self.current.shape

        # 生成空间频率网格
        fx = torch.fft.fftfreq(width, d=self.pixel_size, device=device)
        fy = torch.fft.fftfreq(height, d=self.pixel_size, device=device)
        fx, fy = torch.meshgrid(fx, fy, indexing='xy')

        z = torch.linspace(self.z_start, self.z_end,
                           int((self.z_end - self.z_start) / self.z_step) + 2,
                           device=device, dtype=torch.float32)

        for i in range(len(z)):
            # 严格遵循参考代码的传播项计算
            term = 1.0 - (self.wavelength ** 2) * (fx ** 2 + fy ** 2)
            sqrt_term = torch.sqrt(torch.clamp(term, min=0))  # 负数直接置零
            mask = (term >= 0).type(torch.complex64)

            # 传递函数修正（核心修改）
            H = torch.exp(1j * (2 * torch.pi / self.wavelength) * z[i] * sqrt_term) * mask

            # FFT流程
            U1 = torch.fft.fft2(self.current)
            U2 = U1 * H

            # 动态范围控制优化
            max_val = torch.max(torch.abs(U2))
            if max_val > 1e3:
                U2 = U2 / (max_val / 1e3)

            # 重建与归一化
            recon = torch.abs(torch.fft.ifft2(U2))
            recon = (recon / torch.max(recon + 1e-7)).cpu().numpy()
            self.reconstruction_list.append(recon)

            # plt.imsave(
            #     f'F:/dongjiayao/Data/HoloLab_testData/reconstruction/output/{i}plane_{z[i]:.10f}.jpg',
            #     recon,
            #     cmap='gray'
            # )

        return None


if __name__ == '__main__':
    print('Utils Reconstruction Module', end='\n\n')

    '''
        预定义一个全息，并加载模拟数据，用于测试，包括
        - 原始全息图
        - 预处理后的图
        - 重建完成的图
        - 聚焦完成的图
        - 波前和相位
    '''
    hologram = MockData()
    print(f'- Hologram Data loaded successfully: {hologram.data_path}')
