from module.hologram import Hologram
from module.config import HoloConfig
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import os
import cv2
import numpy as np

try:
    # import cupy as cp
    import torch
except Exception as e:
    print(f"[Module] {e}")

class Polarization_LZM():
    def __init__(self, hologram=None, config=None):

        self.hologram      = hologram.hologram
        self.hologram_p000 = hologram.hologram_p000
        self.hologram_p045 = hologram.hologram_p045
        self.hologram_p090 = hologram.hologram_p090
        self.hologram_p135 = hologram.hologram_p135

        self.AOP           = hologram.AoP
        self.DOLP          = hologram.DoLP

    def run(self):
        return self.hologram_p000, self.hologram_p045, self.hologram_p090, self.hologram_p135

    def Polarization_Separate(self):
        a = 1

    def Calculation_AOP(self):
        a = 1

    def Calculation_DOLP(self):
        a = 1

    def Calculation_Others(self):
        a = 1

class Polarization_():
    def __init__(self, hologram: Hologram, config: HoloConfig):
        """
        初始化偏振处理类

        参数:
        focus: 聚焦后的全息图像数据（numpy数组）
        split_mode: 图像拆分模式
        """
        self.hologram_instance  = hologram  # 保存 hologram 实例以便后续赋值
        # 输入图像为 hologram.focusing，它被视为复合偏振图像
        self.composite_image    = hologram.focusing
        self.split_mode         = config.polarization['split_mode']
        # 检查输入图像
        if not isinstance(self.composite_image, np.ndarray) or self.composite_image.size == 0:
            hologram.status_msg = '输入到 Polarization 的 hologram.focusing 图像无效或为空。'
            raise ValueError("输入到 Polarization 的 hologram.focusing 图像无效或为空。")

        # --- 确定默认空图像的形状 ---
        # 偏振子图的形状是复合图像尺寸的一半
        if isinstance(self.composite_image, np.ndarray) and self.composite_image.ndim >= 2:
            comp_h, comp_w = self.composite_image.shape[:2]
            self._default_sub_img_shape = (comp_h // 2, comp_w // 2)

        # --- 初始化内部属性为安全的空图像默认值 ---
        # 拆分后的四个偏振方向的图像：0°, 45°, 90°, 135° (作为中间变量)
        self._I0 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._I45 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._I90 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._I135 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)

        # 斯托克斯参数（整张图像）(作为中间变量)
        self._S0 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._S1 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._S2 = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)

        self._DoLP = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)  # uint8
        self._AoP = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)  # uint8
        self._total_amp = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)

    def run(self):
        """根据指定模式拆分偏振图像，计算偏振参数，并回传结果"""
        if   self.split_mode == 'quadrant':
            self._split_quadrant()  # 四象限模式
        elif self.split_mode == 'super_pixel':
            self._split_superpixel()  # 超像素模式
        else:
            raise ValueError(f"无效的拆分模式: {self.split_mode}。请使用 'quadrant' 或 'super_pixel'。")

        self._calculate_polarization()
        self.modify_hologram_and_config()

    def modify_hologram_and_config(self):
        if self.hologram_instance is not None:
            # 赋值拆分后的四个偏振子图
            self.hologram_instance.hologram_p000 = self._I0
            self.hologram_instance.hologram_p045 = self._I45
            self.hologram_instance.hologram_p090 = self._I90
            self.hologram_instance.hologram_p135 = self._I135

            # 赋值计算得到的整图偏振参数
            self.hologram_instance.S0 = self._S0
            self.hologram_instance.S1 = self._S1
            self.hologram_instance.S2 = self._S2
            self.hologram_instance.total_amp = self._total_amp  # 定义为 sqrt(S1^2+S2^2)
            self.hologram_instance.DoLP = self._DoLP  # 整张图的 DoLP 图像
            self.hologram_instance.AoP = self._AoP  # 整张图的 AoP 图像
            # print("偏振处理完成，结果已赋值回 hologram 对象。")
        else:
            print("警告: hologram 实例在 Polarization 中为 None，无法回传偏振处理结果!")

    def _split_quadrant(self):
        """四象限拆分：将图像按四个区域划分为四个偏振方向"""
        h, w = self.composite_image.shape[:2]  # 支持灰度和彩色图的拆分
        if h % 2 != 0 or w % 2 != 0:
            print(f"警告: 四象限拆分图像尺寸 ({h}x{w}) 不是偶数，可能导致拆分不精确。将进行取整。")

        h_half, w_half = h // 2, w // 2

        # 确保使用 .copy() 以免后续修改影响原始数据或 hologram 实例中的其他引用
        self._I0 = self.composite_image[0:h_half, 0:w_half].copy()
        self._I45 = self.composite_image[0:h_half, w_half:w_half * 2].copy()  # w_half*2 确保取到末尾
        self._I90 = self.composite_image[h_half:h_half * 2, w_half:w_half * 2].copy()
        self._I135 = self.composite_image[h_half:h_half * 2, 0:w_half].copy()

    def _split_superpixel(self):
        """超像素拆分：2×2 超像素块映射为四个偏振方向"""
        h, w = self.composite_image.shape
        self._I0 = np.zeros((h // 2, w // 2), dtype=np.float32)
        self._I45 = np.zeros_like(self._I0)
        self._I90 = np.zeros_like(self._I0)
        self._I135 = np.zeros_like(self._I0)

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                if i + 1 < h and j + 1 < w:
                    block = self.composite_image[i:i + 2, j:j + 2]
                    self._I0[i // 2, j // 2] = block[0, 0]     # 左上
                    self._I45[i // 2, j // 2] = block[0, 1]    # 右上
                    self._I90[i // 2, j // 2] = block[1, 1]    # 右下
                    self._I135[i // 2, j // 2] = block[1, 0]   # 左下

    def _calculate_polarization(self):
        """计算斯托克斯参数与偏振参数（含归一化和数据转换）"""
        # 确保子图已生成
        if self._I0 is None:  # 任意一个为None即可判断
            raise RuntimeError("在计算偏振参数之前必须先拆分图像。")

        # torch 设备：有 CUDA 用 GPU，否则 CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将 NumPy 数组（可能是 uint8）转换为 float32 以进行精确计算，然后移至 torch
        I0_f = torch.as_tensor(self._I0.astype(np.float32), device=device)
        I45_f = torch.as_tensor(self._I45.astype(np.float32), device=device)
        I90_f = torch.as_tensor(self._I90.astype(np.float32), device=device)
        I135_f = torch.as_tensor(self._I135.astype(np.float32), device=device)

        # 计算斯托克斯参数
        S0_calc = I0_f + I90_f
        S1_calc = I0_f - I90_f
        S2_calc = I45_f - I135_f

        # 幅度信息（S1 和 S2 的模）
        total_amp_calc = torch.sqrt(S1_calc ** 2 + S2_calc ** 2)

        # 计算偏振度 DoLP（归一化并转换为灰度图像）
        epsilon = 1e-10
        DoLP_calc = torch.sqrt(S1_calc ** 2 + S2_calc ** 2) / (S0_calc + epsilon)
        DoLP_calc = torch.clamp(DoLP_calc, 0, 1)  # DoLP 范围在 0 到 1
        DoLP_gray = (DoLP_calc * 255).to(torch.uint8)

        # 计算偏振角 AoP（AoP ∈ [-π/2, π/2]，并归一化为 0~255）
        AoP_rad = 0.5 * torch.arctan2(S2_calc, S1_calc + epsilon)  # AoP 范围在 -pi/2 到 pi/2
        # 将 AoP 从 [-π/2, π/2] 映射到 [0, 1] 区间
        AoP_norm = (AoP_rad + (torch.pi / 2)) / torch.pi
        # 转成 0–255 uint8 图像
        AoP_gray = (AoP_norm * 255).to(torch.uint8)

        # 将计算结果（仍为 torch tensor）赋值给内部变量
        self._S0 = S0_calc
        self._S1 = S1_calc
        self._S2 = S2_calc
        self._DoLP = DoLP_gray
        self._AoP = AoP_gray
        self._total_amp = total_amp_calc

        # 将 torch tensor 转回 NumPy 数组以便赋值给 hologram (假设 hologram 属性是 NumPy)
        self._I0 = I0_f.to(self._I0.dtype).cpu().numpy()
        self._I45 = I45_f.to(self._I45.dtype).cpu().numpy()
        self._I90 = I90_f.to(self._I90.dtype).cpu().numpy()
        self._I135 = I135_f.to(self._I135.dtype).cpu().numpy()

        self._S0 = self._S0.cpu().numpy()
        self._S1 = self._S1.cpu().numpy()
        self._S2 = self._S2.cpu().numpy()
        self._total_amp = self._total_amp.cpu().numpy()
        self._DoLP = self._DoLP.cpu().numpy()
        self._AoP = self._AoP.cpu().numpy()

    # '''---------------- CuPy 备份版本（保留以备需要时切换）----------------
    # 注意：cupy 与 torch 不可混用，会导致堆内存损坏 (0xC0000374)。
    # def _calculate_polarization(self):
    #     """计算斯托克斯参数与偏振参数（含归一化和数据转换）"""
    #     # 确保子图已生成
    #     if self._I0 is None:  # 任意一个为None即可判断
    #         raise RuntimeError("在计算偏振参数之前必须先拆分图像。")
    #
    #     # 将 NumPy 数组（可能是 uint8）转换为 float32 以进行精确计算，然后移至 CuPy
    #     I0_f = cp.asarray(self._I0.astype(np.float32))
    #     I45_f = cp.asarray(self._I45.astype(np.float32))
    #     I90_f = cp.asarray(self._I90.astype(np.float32))
    #     I135_f = cp.asarray(self._I135.astype(np.float32))
    #
    #     # 计算斯托克斯参数
    #     S0_calc = I0_f + I90_f
    #     S1_calc = I0_f - I90_f
    #     S2_calc = I45_f - I135_f
    #
    #     # 幅度信息（S1 和 S2 的模）
    #     total_amp_calc = cp.sqrt(S1_calc ** 2 + S2_calc ** 2)
    #
    #     # 计算偏振度 DoLP（归一化并转换为灰度图像）
    #     epsilon = 1e-10
    #     DoLP_calc = cp.sqrt(S1_calc ** 2 + S2_calc ** 2) / (S0_calc + epsilon)
    #     DoLP_calc = cp.clip(DoLP_calc, 0, 1)  # DoLP 范围在 0 到 1
    #     DoLP_gray = (DoLP_calc * 255).astype(cp.uint8)
    #
    #     # 计算偏振角 AoP（AoP ∈ [-π/2, π/2]，并归一化为 0~255）
    #     AoP_rad = 0.5 * cp.arctan2(S2_calc, S1_calc + epsilon)  # AoP 范围在 -pi/2 到 pi/2
    #     # 将 AoP 从 [-π/2, π/2] 映射到 [0, 1] 区间
    #     AoP_norm = (AoP_rad + (cp.pi / 2)) / cp.pi
    #     # 转成 0–255 uint8 图像
    #     AoP_gray = (AoP_norm * 255).astype(cp.uint8)
    #
    #     # 将计算结果（仍为 CuPy 数组）赋值给内部变量
    #     self._S0 = S0_calc
    #     self._S1 = S1_calc
    #     self._S2 = S2_calc
    #     self._DoLP = DoLP_gray
    #     self._AoP = AoP_gray
    #     self._total_amp = total_amp_calc
    #
    #     # 将 CuPy 数组转回 NumPy 数组以便赋值给 hologram (假设 hologram 属性是 NumPy)
    #     self._I0 = cp.asnumpy(I0_f.astype(self._I0.dtype))
    #     self._I45 = cp.asnumpy(I45_f.astype(self._I45.dtype))
    #     self._I90 = cp.asnumpy(I90_f.astype(self._I90.dtype))
    #     self._I135 = cp.asnumpy(I135_f.astype(self._I135.dtype))
    #
    #     self._S0 = cp.asnumpy(self._S0)
    #     self._S1 = cp.asnumpy(self._S1)
    #     self._S2 = cp.asnumpy(self._S2)
    #     self._total_amp = cp.asnumpy(self._total_amp)
    #     self._DoLP = cp.asnumpy(self._DoLP)
    #     self._AoP = cp.asnumpy(self._AoP)
    #     print("偏振参数计算完成。")
    # -------------------------------------------------------------------'''

class Polarization():
    def __init__(self, hologram: Hologram, config: HoloConfig):

        self.hologram_instance  = hologram
        self.composite_image    = hologram.hologram_raw

        self._config             = config
        self.polar_coeff        = config.polarization['polar_coeff']    # 是否计算高阶参数
        self.split_image        = config.polarization['split_image']    # 是否分割图像
        self.split_mode         = config.polarization['split_mode']
        self.device             = config.polarization['device']

        # 偏振子图的形状是复合图像尺寸的一半
        if isinstance(self.composite_image, np.ndarray) and self.composite_image.ndim >= 2:
            comp_h, comp_w = self.composite_image.shape[:2]
            self._default_sub_img_shape = (comp_h // 2, comp_w // 2)

        # 拆分后的四个偏振方向的图像：0°, 45°, 90°, 135° (作为中间变量)
        self._I0    = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._I45   = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._I90   = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._I135  = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)

        # 斯托克斯参数（整张图像）(作为中间变量)
        self._S0    = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._S1    = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)
        self._S2    = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)

        self._DoLP  = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)  # uint8
        self._AoP   = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)  # uint8
        self._total_amp = self.hologram_instance._make_empty_image(shape=self._default_sub_img_shape)

        # 保存操作
        if 1:
            self.save_action    = config.save_and_load['save_polarization']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

    def run(self):
        # 分割图像
        if self.split_image:
            if   self.split_mode == 'quadrant':
                self._split_quadrant()
            elif self.split_mode == 'super_pixel':
                self._split_superpixel()
            else:
                raise ValueError(f"Unknown: {self.split_mode}. Using 'quadrant' or 'super_pixel'。")

            self._refresh_hologram()

        # 计算参数
        if self.polar_coeff:
            if self.device == 'gpu':
                self._calculate_polarization_gpu()
            else:
                self._calculate_polarization_cpu()

        self.modify_hologram_and_config()
        self.save_to_file()

    def modify_hologram_and_config(self):
        if self.hologram_instance is not None:
            # 赋值拆分后的四个偏振子图
            self.hologram_instance.hologram_p000 = self._I0
            self.hologram_instance.hologram_p045 = self._I45
            self.hologram_instance.hologram_p090 = self._I90
            self.hologram_instance.hologram_p135 = self._I135

            # 偏振参量
            self.hologram_instance.polar_S0      = self._S0
            self.hologram_instance.polar_S1      = self._S1
            self.hologram_instance.polar_S2      = self._S2
            self.hologram_instance.polar_amp     = self._total_amp  # 定义为 sqrt(S1^2+S2^2)
            self.hologram_instance.DoLP          = self._DoLP       # 整张图的 DoLP 图像
            self.hologram_instance.AoP           = self._AoP        # 整张图的 AoP 图像

            self.hologram_instance.status_msg    = "Polarization Separated, Calculated Done."
        else:
            print("Wrong, Polarization is None")

    def save_to_file(self):
        if self.creat_sub_dir:
            save_path = os.path.join(self.save_path, self.image_name)
            save_path = os.path.join(save_path, 'Polarization')
        else:
            save_path = os.path.join(self.save_path, 'Polarization')

        if self.save_action:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if self.split_image:
                sub_images = [self._I0, self._I45, self._I90, self._I135]
                sub_names  = ['I_000', 'I_045', 'I_090', 'I_135']
                for sub_image, sub_name in zip(sub_images, sub_names):
                    save_name       = f"{sub_name}.png"
                    save_path_file  = os.path.join(save_path, save_name)

                    normalized      = (sub_image - np.min(sub_image)) / (np.max(sub_image) - np.min(sub_image) + 1e-10)
                    sub_image       = (normalized * 255).astype(np.uint8)  # 转为 [0, 255] 的 uint8
                    cv2.imwrite(save_path_file, sub_image)

            if self.polar_coeff:
                polar_coeffs = [self._S0, self._S1, self._S2, self._DoLP, self._AoP, self._total_amp]
                polar_names  = ['Polar_S0', 'Polar_S1', 'Polar_S2', 'Polar_DoLP', 'Polar_AoP', 'Polar_Total_Amp']
                for polar_image, polar_name in zip(polar_coeffs, polar_names):
                    save_name       = f"{polar_name}.png"
                    save_path_file  = os.path.join(save_path, save_name)

                    normalized      = (polar_image - np.min(polar_image)) / (np.max(polar_image) - np.min(polar_image) + 1e-10)
                    polar_image     = (normalized * 255).astype(np.uint8)  # 转为 [0, 255] 的 uint8
                    cv2.imwrite(save_path_file, polar_image)

    def _split_quadrant(self):
        """四象限拆分：将图像按四个区域划分为四个偏振方向"""
        h, w = self.composite_image.shape[:2]  # 支持灰度和彩色图的拆分
        if h % 2 != 0 or w % 2 != 0:
            print(f"警告: 四象限拆分图像尺寸 ({h}x{w}) 不是偶数，可能导致拆分不精确。将进行取整。")

        h_half, w_half = h // 2, w // 2

        self._I0    = self.composite_image[0:h_half, 0:w_half].copy()
        self._I45   = self.composite_image[0:h_half, w_half:w_half * 2].copy()  # w_half*2 确保取到末尾
        self._I90   = self.composite_image[h_half:h_half * 2, w_half:w_half * 2].copy()
        self._I135  = self.composite_image[h_half:h_half * 2, 0:w_half].copy()

    def _split_superpixel(self):
        """超像素拆分：2×2 超像素块映射为四个偏振方向"""
        h, w = self.composite_image.shape
        self._I0    = np.zeros((h // 2, w // 2), dtype=np.float32)
        self._I45   = np.zeros_like(self._I0)
        self._I90   = np.zeros_like(self._I0)
        self._I135  = np.zeros_like(self._I0)

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                if i + 1 < h and j + 1 < w:
                    block = self.composite_image[i:i + 2, j:j + 2]
                    self._I0[i // 2, j // 2]    = block[0, 0]     # 左上
                    self._I45[i // 2, j // 2]   = block[0, 1]    # 右上
                    self._I90[i // 2, j // 2]   = block[1, 1]    # 右下
                    self._I135[i // 2, j // 2]  = block[1, 0]   # 左下

    def _refresh_hologram(self):
        self.hologram_instance.hologram         = self._I0
        self.hologram_instance.status_msg       = f'Updated Hologram by Polarization.'

        self.image_height, self.image_width     = self._I0.shape
        self._config.image_info['pixel_num_x']  = self.image_height
        self._config.image_info['pixel_num_y']  = self.image_width

    def _calculate_polarization_gpu(self):
        """计算斯托克斯参数与偏振参数（含归一化和数据转换）torch GPU 版本"""

        # torch 设备：有 CUDA 用 GPU，否则 CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将 NumPy 数组（可能是 uint8）转换为 float32 以进行精确计算，然后移至 torch
        I0_f    = torch.as_tensor(self._I0.astype(np.float32), device=device)
        I45_f   = torch.as_tensor(self._I45.astype(np.float32), device=device)
        I90_f   = torch.as_tensor(self._I90.astype(np.float32), device=device)
        I135_f  = torch.as_tensor(self._I135.astype(np.float32), device=device)

        # 计算斯托克斯参数
        S0_calc = I0_f + I90_f
        S1_calc = I0_f - I90_f
        S2_calc = I45_f - I135_f

        # 幅度信息（S1 和 S2 的模）
        total_amp_calc = torch.sqrt(S1_calc ** 2 + S2_calc ** 2)

        # 计算偏振度 DoLP（归一化并转换为灰度图像）
        epsilon     = 1e-10
        DoLP_calc   = torch.sqrt(S1_calc ** 2 + S2_calc ** 2) / (S0_calc + epsilon)
        DoLP_calc   = torch.clamp(DoLP_calc, 0, 1)  # DoLP 范围在 0 到 1
        DoLP_gray   = (DoLP_calc * 255).to(torch.uint8)

        # 计算偏振角 AoP（AoP ∈ [-π/2, π/2]，并归一化为 0~255）
        AoP_rad = 0.5 * torch.arctan2(S2_calc, S1_calc + epsilon)  # AoP 范围在 -pi/2 到 pi/2
        # 将 AoP 从 [-π/2, π/2] 映射到 [0, 1] 区间
        AoP_norm = (AoP_rad + (torch.pi / 2)) / torch.pi
        # 转成 0–255 uint8 图像
        AoP_gray = (AoP_norm * 255).to(torch.uint8)

        self._S0    = S0_calc
        self._S1    = S1_calc
        self._S2    = S2_calc
        self._DoLP  = DoLP_gray
        self._AoP   = AoP_gray
        self._total_amp = total_amp_calc

        # 将 torch tensor 转回 NumPy 数组以便赋值给 hologram (假设 hologram 属性是 NumPy)
        self._I0        = I0_f.to(self._I0.dtype).cpu().numpy()
        self._I45       = I45_f.to(self._I45.dtype).cpu().numpy()
        self._I90       = I90_f.to(self._I90.dtype).cpu().numpy()
        self._I135      = I135_f.to(self._I135.dtype).cpu().numpy()

        self._S0        = self._S0.cpu().numpy()
        self._S1        = self._S1.cpu().numpy()
        self._S2        = self._S2.cpu().numpy()
        self._total_amp = self._total_amp.cpu().numpy()
        self._DoLP      = self._DoLP.cpu().numpy()
        self._AoP       = self._AoP.cpu().numpy()

    # '''---------------- CuPy 备份版本（保留以备需要时切换）----------------
    # 注意：cupy 与 torch 不可混用，会导致堆内存损坏 (0xC0000374)。
    # def _calculate_polarization_gpu(self):
    #     """计算斯托克斯参数与偏振参数（含归一化和数据转换）"""
    #
    #     # 将 NumPy 数组（可能是 uint8）转换为 float32 以进行精确计算，然后移至 CuPy
    #     I0_f    = cp.asarray(self._I0.astype(np.float32))
    #     I45_f   = cp.asarray(self._I45.astype(np.float32))
    #     I90_f   = cp.asarray(self._I90.astype(np.float32))
    #     I135_f  = cp.asarray(self._I135.astype(np.float32))
    #
    #     # 计算斯托克斯参数
    #     S0_calc = I0_f + I90_f
    #     S1_calc = I0_f - I90_f
    #     S2_calc = I45_f - I135_f
    #
    #     # 幅度信息（S1 和 S2 的模）
    #     total_amp_calc = cp.sqrt(S1_calc ** 2 + S2_calc ** 2)
    #
    #     # 计算偏振度 DoLP（归一化并转换为灰度图像）
    #     epsilon     = 1e-10
    #     DoLP_calc   = cp.sqrt(S1_calc ** 2 + S2_calc ** 2) / (S0_calc + epsilon)
    #     DoLP_calc   = cp.clip(DoLP_calc, 0, 1)  # DoLP 范围在 0 到 1
    #     DoLP_gray   = (DoLP_calc * 255).astype(cp.uint8)
    #
    #     # 计算偏振角 AoP（AoP ∈ [-π/2, π/2]，并归一化为 0~255）
    #     AoP_rad = 0.5 * cp.arctan2(S2_calc, S1_calc + epsilon)  # AoP 范围在 -pi/2 到 pi/2
    #     # 将 AoP 从 [-π/2, π/2] 映射到 [0, 1] 区间
    #     AoP_norm = (AoP_rad + (cp.pi / 2)) / cp.pi
    #     # 转成 0–255 uint8 图像
    #     AoP_gray = (AoP_norm * 255).astype(cp.uint8)
    #
    #     self._S0    = S0_calc
    #     self._S1    = S1_calc
    #     self._S2    = S2_calc
    #     self._DoLP  = DoLP_gray
    #     self._AoP   = AoP_gray
    #     self._total_amp = total_amp_calc
    #
    #     # 将 CuPy 数组转回 NumPy 数组以便赋值给 hologram (假设 hologram 属性是 NumPy)
    #     self._I0        = cp.asnumpy(I0_f.astype(self._I0.dtype))
    #     self._I45       = cp.asnumpy(I45_f.astype(self._I45.dtype))
    #     self._I90       = cp.asnumpy(I90_f.astype(self._I90.dtype))
    #     self._I135      = cp.asnumpy(I135_f.astype(self._I135.dtype))
    #
    #     self._S0        = cp.asnumpy(self._S0)
    #     self._S1        = cp.asnumpy(self._S1)
    #     self._S2        = cp.asnumpy(self._S2)
    #     self._total_amp = cp.asnumpy(self._total_amp)
    #     self._DoLP      = cp.asnumpy(self._DoLP)
    #     self._AoP       = cp.asnumpy(self._AoP)
    # -------------------------------------------------------------------'''

    def _calculate_polarization_cpu(self):
        """计算斯托克斯参数与偏振参数（含归一化和数据转换）CPU版本"""

        # 将数组转换为 float32 以进行精确计算
        I0_f    = self._I0.astype(np.float32)
        I45_f   = self._I45.astype(np.float32)
        I90_f   = self._I90.astype(np.float32)
        I135_f  = self._I135.astype(np.float32)

        # 计算斯托克斯参数
        S0_calc = I0_f + I90_f
        S1_calc = I0_f - I90_f
        S2_calc = I45_f - I135_f

        # 幅度信息（S1 和 S2 的模）
        total_amp_calc = np.sqrt(S1_calc ** 2 + S2_calc ** 2)

        # 计算偏振度 DoLP（归一化并转换为灰度图像）
        epsilon = 1e-10
        DoLP_calc = np.sqrt(S1_calc ** 2 + S2_calc ** 2) / (S0_calc + epsilon)
        DoLP_calc = np.clip(DoLP_calc, 0, 1)  # DoLP 范围在 0 到 1
        DoLP_gray = (DoLP_calc * 255).astype(np.uint8)

        # 计算偏振角 AoP（AoP ∈ [-π/2, π/2]，并归一化为 0~255）
        AoP_rad = 0.5 * np.arctan2(S2_calc, S1_calc + epsilon)  # AoP 范围在 -pi/2 到 pi/2
        # 将 AoP 从 [-π/2, π/2] 映射到 [0, 1] 区间
        AoP_norm = (AoP_rad + (np.pi / 2)) / np.pi
        # 转成 0–255 uint8 图像
        AoP_gray = (AoP_norm * 255).astype(np.uint8)

        # 存储计算结果
        self._S0    = S0_calc
        self._S1    = S1_calc
        self._S2    = S2_calc
        self._DoLP  = DoLP_gray
        self._AoP   = AoP_gray
        self._total_amp = total_amp_calc

        # 将处理后的数据转换回原始数据类型（如果需要）
        self._I0    = I0_f.astype(self._I0.dtype)
        self._I45   = I45_f.astype(self._I45.dtype)
        self._I90   = I90_f.astype(self._I90.dtype)
        self._I135  = I135_f.astype(self._I135.dtype)

if __name__ == '__main__':
    print('Utils Polarization Module', end='\n\n')

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