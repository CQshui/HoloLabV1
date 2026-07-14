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

class Reconstruction_():
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

        # 重建缩放因子：在频域裁剪低频以加速重建（与 rcf_scale 一致）
        self.rcf_scale = config.focusing.get('rcf_scale', 8)

        # 保存全尺寸信息用于最终 resize 回原图尺寸
        self._M_full, self._N_full = self.spectrum.shape

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
        '''新增：每一个截面的z'''
        self.reconstruction_z = []
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

        # 将小尺寸重建结果 resize 回全尺寸，保证下游代码兼容
        full_size = (self._N_full, self._M_full)
        for z, image in zip(self.z_list, self.reconstruction_list):
            abs_v = np.abs(image)
            # 归一化
            if abs_v.max() > abs_v.min():
                normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
            else:
                normalized = abs_v
            img_uint8 = (normalized * 255).astype(np.uint8)
            img_full = cv2.resize(img_uint8, full_size, interpolation=cv2.INTER_LINEAR)
            key_z = f"Z {z / unit:.3f} {unit_str}"
            self.reconstruction_dict[key_z] = img_full
            self.reconstruction_z.append(z)

        self._hologram.reconstruction = self.reconstruction_dict
        self._hologram.reconstruction_z = self.reconstruction_z

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
        """预计算波前，并对频谱做低频裁剪以实现缩小重建"""
        scale = self.rcf_scale
        M_full, N_full = self.spectrum.shape
        # 裁剪频谱中心低频区域（等效于缩小空间域图像）
        M_crop = M_full // scale
        N_crop = N_full // scale
        M_start = (M_full - M_crop) // 2
        N_start = (N_full - N_crop) // 2
        self.spectrum_cropped = self.spectrum[M_start:M_start + M_crop, N_start:N_start + N_crop]
        # 缩小后的像素尺寸
        self.pixel_size_scaled = self.pixel_size * scale
        # 波前（从裁剪后的频谱逆变换）
        self.wavefront = np.fft.ifft2(np.fft.ifftshift(self.spectrum_cropped))

    def _precompute_frequency(self):
        M, N = self.wavefront.shape
        self.fft_x = np.fft.fftshift(np.fft.fftfreq(N, d=self.pixel_size_scaled))
        self.fft_y = np.fft.fftshift(np.fft.fftfreq(M, d=self.pixel_size_scaled))
        self.fft_mesh_x, self.fft_mesh_y = np.meshgrid(self.fft_x, self.fft_y)

    def _Free_Space_Propagation_CPU(self, spectrum, z, wavelength, fft_squa):
        H = np.exp(1j * 2 * np.pi / wavelength * z) * \
            np.exp(-1j * np.pi * wavelength * z * fft_squa)

        A = H * spectrum
        U = np.fft.ifft2(np.fft.ifftshift(A))
        return U
    def Reconstruction_Angular_Spectrum_CPU(self):
        spectrum    = self.spectrum_cropped
        wavelength  = self.wavelength
        fft_mesh_x  = self.fft_mesh_x
        fft_mesh_y  = self.fft_mesh_y
        fft_squa    = fft_mesh_x ** 2 + fft_mesh_y ** 2

        for z in self.z_array:
            U  = self._Free_Space_Propagation_CPU(spectrum, z, wavelength, fft_squa)
            self.reconstruction_list.append(U)

    def Reconstruction_Angular_Spectrum_CPU_Multithreading(self):
        results = [None] * len(self.z_array)

        spectrum    = self.spectrum_cropped
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
        """GPU 加速的多截面重建（向量化版本）"""
        import cupy as cp

        # CuPy 数组
        spectrum_gpu = cp.asarray(self.spectrum_cropped)
        wavelength_gpu = cp.asarray(self.wavelength)
        fft_mesh_x_gpu = cp.asarray(self.fft_mesh_x)
        fft_mesh_y_gpu = cp.asarray(self.fft_mesh_y)
        fft_squa_gpu = fft_mesh_x_gpu ** 2 + fft_mesh_y_gpu ** 2

        z_gpu = cp.asarray(self.z_array)  # (Z,)

        # 向量化：z 扩展为 (Z, 1, 1) 与 (M, N) 广播 → (Z, M, N)
        z_reshaped = z_gpu[:, cp.newaxis, cp.newaxis]  # (Z, 1, 1)

        H = cp.exp(1j * 2 * cp.pi / wavelength_gpu * z_reshaped) * \
            cp.exp(-1j * cp.pi * wavelength_gpu * z_reshaped * fft_squa_gpu)  # (Z, M, N)

        # 批量传播：spectrum_gpu (M, N) → (1, M, N) 广播到 (Z, M, N)
        A = H * spectrum_gpu[cp.newaxis, :, :]  # (Z, M, N)
        U = cp.fft.ifft2(cp.fft.ifftshift(A, axes=(1, 2)), axes=(1, 2))  # (Z, M, N)

        # 转回 numpy list
        self.reconstruction_list = [cp.asnumpy(U[i]) for i in range(len(self.z_array))]

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

class Reconstruction():
    def __init__(self, hologram, config):
        # self.hologram_raw   =
        self.creat_sub_dir = None
        self.Holo_class     = hologram  # 整个hologram类
        self.hologram       = hologram.hologram
        self.spectrum       = hologram.spectrum

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

        # 重建缩放因子（与 rcf_scale 一致）
        self.rcf_scale = config.focusing.get('rcf_scale', 8)

        # 对频谱做低频裁剪以加速重建
        scale = self.rcf_scale
        M_full, N_full = self.spectrum.shape
        M_crop = M_full // scale
        N_crop = N_full // scale
        M_start = (M_full - M_crop) // 2
        N_start = (N_full - N_crop) // 2
        self.spectrum_cropped = self.spectrum[M_start:M_start + M_crop, N_start:N_start + N_crop]
        self.pixel_size_scaled = self.pixel_size * scale

        # 保存全尺寸信息用于最终 resize
        self._M_full, self._N_full = M_full, N_full

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

        self.modify_hologram_and_config()
        # self.save_to_file()

    def modify_hologram_and_config(self):
        # 判断z的量级
        unit = 1.0
        if self.z_step >= 1e-4 or (self.z_end - self.z_start) >= 1e-3:
            unit = unit_mm
            unit_str = 'mm'
        else:
            unit = unit_um
            unit_str = 'um'

        # 将小尺寸重建结果 resize 回全尺寸，保证下游代码兼容
        full_size = (self._N_full, self._M_full)
        for z, image in zip(self.z_list, self.reconstruction_list):
            abs_v = np.abs(image)
            if abs_v.max() > abs_v.min():
                normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
            else:
                normalized = abs_v
            img_uint8 = (normalized * 255).astype(np.uint8)
            img_full = cv2.resize(img_uint8, full_size, interpolation=cv2.INTER_LINEAR)
            key_z = f"Z {z / unit:.3f} {unit_str}"
            self.reconstruction_dict[key_z] = img_full

        self.Holo_class.reconstruction = self.reconstruction_dict
        self.Holo_class.reconstruction_z = self.z_list
        self.Holo_class.status_msg = f"Reconstruction Finished. Total {len(self.reconstruction_dict)} Planes"

    def save_to_file(self):
        if self.creat_sub_dir:
            save_path = os.path.join(self.save_path, self.image_name)
            save_path = os.path.join(save_path, 'Reconstruction')
        else:
            save_path = os.path.join(self.save_path, 'Reconstruction')

        if self.save_action:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for _key, _value in self.reconstruction_dict.items():
                abs_v = np.abs(_value.copy())
                normalized = (abs_v - np.min(abs_v)) / (np.max(abs_v) - np.min(abs_v) + 1e-10)
                img_save = (normalized * 255).astype(np.uint8)
                save_name = f"{_key}.png"
                save_path_file = os.path.join(save_path, save_name)
                cv2.imwrite(save_path_file, img_save)

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
            self.current = ifft2(ifftshift(self.spectrum_cropped))

        else:
            print('离轴处理')
            self.current = ifft2(ifftshift(self.spectrum_cropped))

        # 图像尺寸（小尺寸）
        width, height = self.current.shape[1], self.current.shape[0]

        # 空间频率（使用缩放后的像素尺寸）
        fx = np.linspace(-1 / (2 * self.pixel_size_scaled), 1 / (2 * self.pixel_size_scaled), width)
        fy = np.linspace(-1 / (2 * self.pixel_size_scaled), 1 / (2 * self.pixel_size_scaled), height)
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
        """GPU 加速的多截面重建（PyTorch 向量化版本）"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.type == 'Inline':
            print('同轴处理')
        else:
            print('离轴处理')

        spectrum_t = torch.tensor(self.spectrum_cropped, device=device, dtype=torch.complex64)
        self.current = torch.fft.ifft2(torch.fft.ifftshift(spectrum_t))

        height, width = self.current.shape

        # 生成空间频率网格（使用缩放后的像素尺寸）
        fx = torch.fft.fftfreq(width, d=self.pixel_size_scaled, device=device)
        fy = torch.fft.fftfreq(height, d=self.pixel_size_scaled, device=device)
        fx, fy = torch.meshgrid(fx, fy, indexing='xy')

        f_sq = fx ** 2 + fy ** 2  # (H, W)
        term = 1.0 - (self.wavelength ** 2) * f_sq
        sqrt_term = torch.sqrt(torch.clamp(term, min=0))
        mask = (term >= 0).type(torch.complex64)

        # z 向量化
        z = torch.linspace(self.z_start, self.z_end,
                           int((self.z_end - self.z_start) / self.z_step) + 1,
                           device=device, dtype=torch.float32)  # (Z,)

        z_reshaped = z[:, None, None]  # (Z, 1, 1)

        # 传递函数：所有 z 一次计算 → (Z, H, W)
        phi = (2 * torch.pi / self.wavelength) * z_reshaped * sqrt_term
        H = torch.exp(1j * phi) * mask  # (Z, H, W)

        # FFT 一次
        U1 = torch.fft.fft2(self.current)  # (H, W)

        # 批量传播
        U2 = U1 * H  # (Z, H, W)

        # 动态范围控制（取每个平面的最大值）
        max_vals = torch.amax(torch.abs(U2), dim=(1, 2))  # (Z,)
        scale_mask = max_vals > 1e3
        U2[scale_mask] = U2[scale_mask] / (max_vals[scale_mask, None, None] / 1e3)

        # 批量 IFFT
        recon = torch.abs(torch.fft.ifft2(U2))  # (Z, H, W)

        # 归一化 + 转 numpy
        max_per_plane = torch.amax(recon, dim=(1, 2), keepdim=True)  # (Z, 1, 1)
        recon = (recon / (max_per_plane + 1e-7)).cpu().numpy()

        self.reconstruction_list = [recon[i] for i in range(recon.shape[0])]

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