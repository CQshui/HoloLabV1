from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import os
import cv2
import numpy as np
from numpy.fft import fftshift, fft2, ifft2, ifftshift

from functools import partial
import multiprocessing
from multiprocessing import shared_memory, Pool

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt

try:
    import cupy as cp
    import torch
except Exception as e:
    print(f"[Module] {e}")

class Reconstruction():
    def __init__(self, hologram, config):

        self.hologram       = hologram.hologram
        self.spectrum       = hologram.spectrum

        self.pixel_size     = config.image_info['pixel_size'] * unit_um
        self.wavelength     = config.image_info['wavelength'] * unit_nm
        self.pad_factor     = 1.0

        self.method         = config.reconstruction['method']
        self.z_start        = config.reconstruction['z_start'] * unit_mm
        self.z_end          = config.reconstruction['z_end']   * unit_mm
        self.z_step         = config.reconstruction['z_step']  * unit_mm
        self.z_num          = int((self.z_end - self.z_start) / self.z_step) + 1
        self.z_array        = np.linspace(self.z_start, self.z_end, self.z_num)
        self.z_list         = self.z_array.tolist()

        self.cpu_num        = config.reconstruction['cpu_num']
        self.gpu_num        = config.reconstruction['gpu_num']

        if 1:
            self.save_action    = config.save_and_load['save_reconstruction']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

        self._precompute_wavefront()
        self._precompute_frequency()

        '''每一个截面的图像，保存成list'''
        self.reconstruction_list = []
        '''每一个截面的图像，为了便于前端展示，弄成字典形式，key为z，value为图像'''
        self.reconstruction_dict = {}

        '''保留原始对象用于修改'''
        self._hologram = hologram

    def run(self):
        if   self.method == 'Angular_CPU':
            if self.cpu_num > 1:
                # self.Reconstruction_Angular_Spectrum_CPU_Parallel()
                self.Reconstruction_Angular_Spectrum_CPU_Multithreading()
            else:
                self.Reconstruction_Angular_Spectrum_CPU()

        elif self.method == 'Angular_GPU':
            self.Reconstruction_Angular_Spectrum_GPU()

        elif self.method == 'AI':
            pass

        else:
            self.Reconstruction_Example()

        self.modify_hologram_and_config()
        self.save_to_file()

    def modify_hologram_and_config(self):
        # 判断z的量级
        unit = 1.0
        if self.z_step >= 1e-4 or (self.z_end - self.z_start) >= 1e-3:
            unit = unit_mm
            unit_str = 'mm'
        else:
            unit = unit_um
            unit_str = 'um'

        for z, image in zip(self.z_list, self.reconstruction_list):
            key_z = f"Z {z / unit:.3f} {unit_str}"
            self.reconstruction_dict[key_z] = image

        self._hologram.reconstruction = self.reconstruction_dict

        self._hologram.status_msg     = f"Reconstruction Finished. Total {len(self.reconstruction_dict)} Planes"

    def save_to_file(self):

        if self.creat_sub_dir:
            save_path = os.path.join(self.save_path, self.image_name)
            save_path = os.path.join(save_path, 'Reconstruction')
        else:
            save_path = os.path.join(self.save_path, 'Reconstruction')

        if self.save_action:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            reconstruction = self.reconstruction_dict
            for _key, _value in reconstruction.items():
                abs_v = np.abs(_value.copy())
                normalized = (abs_v - np.min(abs_v)) / (np.max(abs_v) - np.min(abs_v) + 1e-10)
                reconstruction[_key] = (normalized * 255).astype(np.uint8)  # 转为 [0, 255] 的 uint8

            for z, image in reconstruction.items():
                save_name       = f"{z}.png"
                save_path_file  = os.path.join(save_path, save_name)
                cv2.imwrite(save_path_file, image)

    def _precompute_wavefront(self):
        self.wavefront = np.fft.ifft2(np.fft.ifftshift(self.spectrum))
    def _precompute_frequency(self):
        M, N = self.wavefront.shape
        self.fft_x = np.fft.fftshift(np.fft.fftfreq(N, d=self.pixel_size))
        self.fft_y = np.fft.fftshift(np.fft.fftfreq(M, d=self.pixel_size))
        self.fft_mesh_x, self.fft_mesh_y = np.meshgrid(self.fft_x, self.fft_y)

    def _Free_Space_Propagation_CPU(self, spectrum, z, wavelength, fft_squa):
        H = np.exp(1j * 2 * np.pi / wavelength * z) * \
            np.exp(-1j * np.pi * wavelength * z * fft_squa)

        A = H * spectrum
        U = np.fft.ifft2(np.fft.ifftshift(A))
        return U
    def Reconstruction_Angular_Spectrum_CPU(self):
        spectrum    = self.spectrum
        wavelength  = self.wavelength
        fft_mesh_x  = self.fft_mesh_x
        fft_mesh_y  = self.fft_mesh_y
        fft_squa    = fft_mesh_x ** 2 + fft_mesh_y ** 2

        for z in self.z_array:
            U  = self._Free_Space_Propagation_CPU(spectrum, z, wavelength, fft_squa)
            self.reconstruction_list.append(U)
    def Reconstruction_Angular_Spectrum_CPU_Multithreading(self):
        results = [None] * len(self.z_array)

        spectrum    = self.spectrum
        wavelength  = self.wavelength
        fft_mesh_x  = self.fft_mesh_x
        fft_mesh_y  = self.fft_mesh_y
        fft_squa    = fft_mesh_x ** 2 + fft_mesh_y ** 2

        def worker(index, z):
            U = self._Free_Space_Propagation_CPU(spectrum, z, wavelength, fft_squa)
            results[index] = U

        # 使用ThreadPoolExecutor管理线程池
        self.thread_num = self.cpu_num
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            # 提交所有任务
            futures = {executor.submit(worker, i, z): i
                       for i, z in enumerate(self.z_array)}

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果（如果有异常会抛出）
                except Exception as e:
                    print(f"Thread execution failed: {e}")

        self.reconstruction_list = results

    def _Free_Space_Propagation_GPU(self, spectrum, z, wavelength, fft_squa):
        H = np.exp(1j * 2 * cp.pi / wavelength * z) * \
            np.exp(-1j * cp.pi * wavelength * z * fft_squa)

        A = H * spectrum
        U = cp.fft.ifft2(np.fft.ifftshift(A))

        return cp.asnumpy(U)
    def Reconstruction_Angular_Spectrum_GPU(self):
        spectrum    = cp.asarray(self.spectrum)
        wavelength  = cp.asarray(self.wavelength)
        fft_mesh_x  = cp.asarray(self.fft_mesh_x)
        fft_mesh_y  = cp.asarray(self.fft_mesh_y)
        fft_squa    = fft_mesh_x ** 2 + fft_mesh_y ** 2

        for z in self.z_array:
            U  = self._Free_Space_Propagation_GPU(spectrum, z, wavelength, fft_squa)
            self.reconstruction_list.append(U)

    '''备份------------------------------------------------------------'''
    def _Free_Space_Propagation_CPU_(self, U0, distance, pad_factor=1):
        wavelength   = self.wavelength
        z            = distance

        M, N         = U0.shape
        M_pad, N_pad = int(M * pad_factor), int(N * pad_factor)

        # 1. 零填充
        U0_padded    = np.zeros((M_pad, N_pad), dtype=complex)
        M_start, N_start = (M_pad - M) // 2, (N_pad - N) // 2
        U0_padded[M_start:M_start + M, N_start:N_start + N] = U0

        # 2. 新的频域坐标
        fft_x = np.fft.fftshift(np.fft.fftfreq(N_pad, d=self.pixel_size))
        fft_y = np.fft.fftshift(np.fft.fftfreq(M_pad, d=self.pixel_size))
        fft_mesh_x, fft_mesh_y = np.meshgrid(fft_x, fft_y)

        # 3. 传播函数
        H = np.exp(1j * 2 * np.pi / wavelength * z) * \
            np.exp(-1j * np.pi * wavelength * z * (fft_mesh_x ** 2 + fft_mesh_y ** 2))

        # 4. 傅里叶传播
        A0 = np.fft.fftshift(np.fft.fft2(U0_padded))
        A = H * A0
        U_padded = np.fft.ifft2(np.fft.ifftshift(A))

        # 5. 裁剪回原大小
        U = U_padded[M_start:M_start + M, N_start:N_start + N]

        return U
    def Reconstruction_Angular_Spectrum_CPU_(self):
        U0 = np.fft.ifft2(np.fft.ifftshift(self.spectrum))
        for z in self.z_array:
            U  = self._Free_Space_Propagation_CPU(U0, z)
            self.reconstruction_list.append(U)
    def Reconstruction_Angular_Spectrum_CPU_Multithreading_(self):
        """CPU多线程角谱重建"""
        U0 = np.fft.ifft2(np.fft.ifftshift(self.spectrum))

        # 预分配结果列表，保持z的顺序
        results = [None] * len(self.z_array)

        def worker(index, z):
            """线程工作函数"""
            U = self._Free_Space_Propagation_CPU(U0, z)
            results[index] = U

        # 使用ThreadPoolExecutor管理线程池
        self.thread_num = self.cpu_num
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            # 提交所有任务
            futures = {executor.submit(worker, i, z): i
                       for i, z in enumerate(self.z_array)}

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果（如果有异常会抛出）
                except Exception as e:
                    print(f"Thread execution failed: {e}")

        self.reconstruction_list = results

    def _Free_Space_Propagation_GPU_(self, U0, distance, pad_factor=1):
        wavelength = self.wavelength
        z = distance
        pixel_size = self.pixel_size

        # 将输入转换为 cupy 数组
        U0_cp = cp.asarray(U0)

        M, N = U0.shape
        M_pad, N_pad = int(M * pad_factor), int(N * pad_factor)

        # 1. 零填充
        U0_padded = cp.zeros((M_pad, N_pad), dtype=cp.complex64)
        M_start, N_start = (M_pad - M) // 2, (N_pad - N) // 2
        U0_padded[M_start:M_start + M, N_start:N_start + N] = U0_cp

        # 2. 新的频域坐标（仍然用 numpy 计算，然后转 cp）
        fft_x = cp.fft.fftshift(cp.fft.fftfreq(N_pad, d=pixel_size))
        fft_y = cp.fft.fftshift(cp.fft.fftfreq(M_pad, d=pixel_size))
        fft_mesh_x, fft_mesh_y = cp.meshgrid(fft_x, fft_y)

        # 3. 传播函数
        H = cp.exp(1j * 2 * cp.pi / wavelength * z) * \
            cp.exp(-1j * cp.pi * wavelength * z * (fft_mesh_x ** 2 + fft_mesh_y ** 2))

        # 4. 傅里叶传播（使用 cupy 的 fft）
        A0 = cp.fft.fftshift(cp.fft.fft2(U0_padded))
        A = H * A0
        U_padded = cp.fft.ifft2(cp.fft.ifftshift(A))

        # 5. 裁剪回原大小
        U = U_padded[M_start:M_start + M, N_start:N_start + N]

        # 转回 numpy 数组
        return cp.asnumpy(U)
    def Reconstruction_Angular_Spectrum_GPU_(self):
        U0 = np.fft.ifft2(np.fft.ifftshift(self.spectrum))
        for z in self.z_array:
            U  = self._Free_Space_Propagation_GPU(U0, z)
            self.reconstruction_list.append(U)
    '''备份____________________________________________________________'''

class Reconstruction_DongJY():
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
    a = 1