from module.hologram import Hologram
from module.config import HoloConfig
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import numpy as np
import time
import cv2
import os
import pywt
from scipy.ndimage import uniform_filter

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset
    from pytorch_wavelets import DWTForward, DWTInverse

    from concurrent.futures import ProcessPoolExecutor  # CPU并行化
    from pathlib import Path

except Exception as e:
    print(f"[Module] {e}")

from utils.utils_for_focusing.yoloV8.yolo import YOLO
from utils.utils_for_focusing.sort.kalman_trace_iou_center import Kalman_tracker, Sort
from utils.utils_for_focusing.rcf.models.RCF import RCF
from utils.utils_for_focusing.rcf.models.RCF_ASPP import RCF_ASPP
from utils.utils_for_focusing.final import RCFLoader, rcf_predict, rcf_predict_V1, rcf_predict_V2
from utils.utils_for_focusing.rcf.data_loader import prepare_image_PIL, convert_to_rgb

from utils.fast_focus import FastFocusPCHIP, FastFocusPCHIP_GPU
from utils.fast_focus_multi import MultiFocusPCHIP, MultiFocusPCHIP_GPU

class Focusing_LZM():
    def __init__(self, hologram, config):

        self.reconstruction = hologram.reconstruction

        self.method         = config.focusing['method']

        self.focusing               = None
        self.focusing_each          = []
        self.focusing_depth_map     = None

    def run(self):
        # if self.method == 'Angular':
        #     pass
        # elif self.method == 'WaveLet':
        #     pass
        # else:
        #     self.AutoFocusing_Example()

        image_url = r'D:\Development\HoloLab\test_data\focusing\test.png'
        image = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)

        self.focusing = image

        return self.focusing

    def AutoFocusing_Example(self):
        self.focusing = np.random.rand(50, 50)
        time.sleep(2)

    def AutoFocusing_WaveLet_CPU(self):
        a = 1
    def AutoFocusing_Gradient_Variance_CPU(self):
        a = 1
    def AutoFocusing_Machine_Learning_CPU(self):
        a = 1
    def AutoFocusing_WaveLet_GPU(self):
        a = 1
    def AutoFocusing_Gradient_Variance_GPU(self):
        a = 1
    def AutoFocusing_Machine_Learning_GPU(self):
        a = 1

'''DongJY@2025.06.03'''
class RCFProcessor:
    def __init__(self, rcf_model, device, stacks, stacks_ori, rcf_scale, z_list):
        self.rcf_model = rcf_model
        self.device = device
        self.stacks = stacks
        self.stacks_ori = stacks_ori
        self.rcf_scale = rcf_scale
        self.z_list = z_list  # 重建时的z序列
        # 注意：这里不再初始化 focusing_z，因为它应该是进程局部的

    def process_stack(self, stack_idx):
        stack = self.stacks[stack_idx]
        stack_ori = self.stacks_ori[stack_idx]

        # 执行预测
        img_tmp, index_tmp = rcf_predict(
            self.rcf_model,
            get_dataset(stack, stack_ori, k_size=self.rcf_scale),
            device=self.device
        )

        # 计算当前stack的聚焦z值
        focusing_z = self.z_list[index_tmp]

        # 返回结果：聚焦后的图像和对应的z值
        return torch.squeeze(img_tmp.squeeze()).cpu().numpy(), focusing_z

class get_dataset():
    def __init__(self, stack, stack_ori, k_size=6):
        self.stack = stack
        self.stack_ori = stack_ori
        self.k_size = k_size
        self.length = len(stack)

    # 启动Dataloader似乎会耗费大量时间，没有必要
    def get_data(self, index):
        img = self.stack[index]
        # img = np.array(self.stack[index], dtype=np.float32)
        img_ori = self.stack_ori[index]
        # img_ori = np.array(self.stack_ori[index], dtype=np.float32)

        img = convert_to_rgb(img)  # 确保图像是RGB格式
        img = cv2.resize(img, (img.shape[1] // self.k_size, img.shape[0] // self.k_size),
                         interpolation=cv2.INTER_NEAREST)
        # img = cv2.resize(img, (224, 224),
        #                  interpolation=cv2.INTER_NEAREST)
        img = prepare_image_PIL(img)

        # 转 tensor: CxHxW, 归一化到 [0,1]
        img = torch.from_numpy(img).float().unsqueeze(0)
        # 原始图直接转 tensor
        img_ori = torch.from_numpy(img_ori).unsqueeze(0).float()

        return img, img_ori

class Focusing:
    _model_loaded = False
    _yolo_model = None
    _rcf_model = None

    def __init__(self, hologram, config, **kwargs):

        self.focusing        = None
        self.focusing_z      = []   # 每个颗粒的聚焦深度位置
        self.focusing_xy     = []   # 每个颗粒的位置框坐标，嵌套列表[[x1,y1,x2,y2],
        self.focusing_each   = {}

        _reconstruction      = hologram.reconstruction
        _reconstruction_list = [_reconstruction[i] for i in _reconstruction.keys()]
        # 处理时用压缩图加速；最终输出用原图分辨率
        if len(_reconstruction_list) > 0:
            h_full, w_full = _reconstruction_list[0].shape[:2]
            self._full_h, self._full_w = h_full, w_full  # 记住原图尺寸
            scale = config.focusing['rcf_scale']
            h_small, w_small = h_full // scale, w_full // scale
            _reconstruction_list = [
                cv2.resize(img, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                for img in _reconstruction_list
            ]
        else:
            self._full_h, self._full_w = None, None
        self.stack           = np.stack(_reconstruction_list, axis=2).astype(np.float64) * 255  # (height, width, num)

        self.method         = kwargs.get('method', config.focusing['method'])
        self.device         = torch.device(config.focusing['device'])
        self.gpu_id         = config.focusing['gpu_id']
        self.cpu_num        = config.focusing['cpu_num']        # cpu线程数

        self.yoloModel_path = config.focusing['yolo_model_path']
        self.rcfModel_path  = config.focusing['rcf_model_path']
        self.rcf_scale      = config.focusing['rcf_scale']      # 用于rcf图像缩放
        self.min_particle_area = config.focusing.get('min_particle_area', 1600)  # AI+Wavelet/Gradient 颗粒最小面积

        self.get_model      = config.focusing['get_model']      # 如果为True，将直接给Focus类传入模型本身，而不是根据路径加载模型

        if 1:
            self.save_action    = config.save_and_load['save_focusing']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

        if 'yoloModel' in kwargs:
            # 检查**kwargs中是否传入了yoloModel参数
            self.yoloModel  = kwargs['yoloModel']
            self.rcfModel   = kwargs['rcfModel']

        elif self.method in ['AI', 'AI_Wavelet', 'AI_Gradient']:
            # AI 方法需要 YOLO + RCF 模型
            if Focusing._yolo_model is None:
                self.load_model()  # 同时加载 YOLO 和 RCF
            else:
                self.yoloModel = Focusing._yolo_model
                self.yoloModel.net.to(self.device)
                self.rcfModel = Focusing._rcf_model
                if self.rcfModel is not None:
                    self.rcfModel.to(self.device)

        elif self.method in ['PCHIP', 'PCHIP_GPU']:
            # PCHIP 方法只需要 RCF 模型，不需要 YOLO
            self.yoloModel = None
            if Focusing._rcf_model is None:
                self.load_rcf_model()
            else:
                self.rcfModel = Focusing._rcf_model
                if self.rcfModel is not None:
                    self.rcfModel.to(self.device)

        elif self.method in ['Multi_PCHIP', 'Multi_PCHIP_GPU']:
            # Multi_PCHIP 内部自行管理模型加载
            self.yoloModel = None
            self.rcfModel = None

        else:
            # Wavelet / Gradient 等方法不需要模型
            self.yoloModel = None
            self.rcfModel = None

        '''保留原始对象用于修改'''
        self._hologram      = hologram
        self._config        = config

    # 执行函数，需要输入方法名称
    def run(self):

        if   self.method == 'Wavelet' and self.device.type == 'cpu':
            result = self.AutoFocusing_WaveLet_CPU()
        elif self.method == 'Wavelet' and self.device.type == 'cuda':
            result = self.AutoFocusing_WaveLet_GPU()

        elif self.method == 'Gradient' and self.device.type == 'cpu':
            result = self.AutoFocusing_Gradient_Variance_CPU()
        elif self.method == 'Gradient' and self.device.type == 'cuda':
            result = self.AutoFocusing_Gradient_Variance_GPU()

        elif self.method == 'AI' and self.device.type == 'cpu':
            self.AutoFocusing_AI_CPU()
        elif self.method == 'AI' and self.device.type == 'cuda':
            self.AutoFocusing_AI_GPU()

        elif self.method == 'AI_Wavelet' and self.device.type == 'cpu':
            result = self.AutoFocusing_Wavelet_AI_CPU()
        elif self.method == 'AI_Wavelet' and self.device.type == 'cuda':
            result = self.AutoFocusing_Wavelet_AI_GPU()

        elif self.method == 'AI_Gradient' and self.device.type == 'cpu':
            result = self.AutoFocusing_Gradient_Variance_AI_CPU()
        elif self.method == 'AI_Gradient' and self.device.type == 'cuda':
            result = self.AutoFocusing_Gradient_Variance_AI_GPU()

        elif self.method == 'PCHIP':
            result = self.AutoFocusing_PCHIP()
        elif self.method == 'PCHIP_GPU':
            result = self.AutoFocusing_PCHIP_GPU()

        elif self.method == 'Multi_PCHIP':
            result = self.AutoFocusing_Multi_PCHIP()
        elif self.method == 'Multi_PCHIP_GPU':
            result = self.AutoFocusing_Multi_PCHIP_GPU()

        else:
            print('No such choice!')
        self.modify_hologram_and_config()
        self.save_to_file()

    def modify_hologram_and_config(self):
        self._hologram.focusing      = self.focusing
        self._hologram.focusing_z    = self.focusing_z
        self._hologram.focusing_xy   = self.focusing_xy
        self._hologram.focusing_each = self.focusing_each
        self._hologram.status_msg    = f"Focusing Finished."

    def save_to_file(self):

        if self.creat_sub_dir:
            save_path = os.path.join(self.save_path, self.image_name)
            save_path = os.path.join(save_path, 'Focusing')
        else:
            save_path = os.path.join(self.save_path, 'Focusing')

        if self.save_action:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # print(f"Save Focusing to {save_path}")
            '''总体'''
            # focusing_norm = (self.focusing - np.min(self.focusing)) / (np.max(abs_v) - np.min(abs_v) + 1e-10)
            save_name       = f"Focusing.png"
            save_path_file  = os.path.join(save_path, save_name)
            cv2.imwrite(save_path_file, self.focusing)

            '''子图'''
            focusing_each = self.focusing_each
            for _key, _value in focusing_each.items():
                abs_v = np.abs(_value.copy())
                normalized = (abs_v - np.min(abs_v)) / (np.max(abs_v) - np.min(abs_v) + 1e-10)
                focusing_each[_key] = (normalized * 255).astype(np.uint8)  # 转为 [0, 255] 的 uint8

            for _name, image in focusing_each.items():
                save_name       = f"{_name}.png"
                save_path_file  = os.path.join(save_path, save_name)
                cv2.imwrite(save_path_file, image)

    # 只加载 RCF 模型（用于 PCHIP 方法），不加载 YOLO
    def load_rcf_model(self):
        """加载 RCF 边缘检测模型（用于 PCHIP 快速自聚焦的聚焦分数计算）"""
        from utils.utils_for_focusing.rcf.models.RCF import RCF
        self.rcfModel = RCF(self.device)
        self.rcfModel.to(self.device)
        checkpoint = torch.load(self.rcfModel_path, map_location=self.device)
        self.rcfModel.load_state_dict(checkpoint['state_dict'])
        Focusing._rcf_model = self.rcfModel

    # 在没有外部传入模型时，根据模型路径加载模型
    def load_model(self):
        if self.get_model:
            pass
        else:
            CUDA_Available = (self.device.type == 'cuda')
            # print(CUDA_Available)

            self.yoloModel = YOLO(input_shape=[640, 640],
                                  phi='s',
                                  model_path=self.yoloModel_path,
                                  cuda=CUDA_Available,
                                  letterbox_image=True,
                                  confidence=0.5,
                                  nms_iou=0.3,
                                  device=self.device)

            # 边缘预测及图像拼接，13截面5颗粒的情况下耗时2s，暂时不做修改；get_data耗费时间和模型预测耗费时间各占一半
            self.rcfModel = RCF(self.device)
            self.rcfModel.to(self.device)
            checkpoint = torch.load(self.rcfModel_path, map_location=self.device)
            self.rcfModel.load_state_dict(checkpoint['state_dict'])

            Focusing._yolo_model = self.yoloModel
            Focusing._rcf_model = self.rcfModel

    def AutoFocusing_WaveLet_CPU(self):
        h, w, num_images = self.stack.shape
        scale = self.rcf_scale
        h_small, w_small = h // scale, w // scale
        stack_small = np.stack([
            cv2.resize(self.stack[:, :, i], (w_small, h_small), interpolation=cv2.INTER_LINEAR)
            for i in range(num_images)
        ], axis=-1)

        wavelet = 'db2'
        window_size = 5
        level = 3

        coeffs_list = []
        for i in range(num_images):
            img = stack_small[:, :, i].copy().astype(np.float32)
            min_size = 2 ** (level + 1)
            pad_h = (min_size - (h_small % min_size)) % min_size
            pad_w = (min_size - (w_small % min_size)) % min_size
            padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='symmetric')
            current_img = padded
            coeffs = []
            for _ in range(level):
                cA, (cH, cV, cD) = pywt.dwt2(current_img, wavelet, mode='periodization')
                coeffs.append((cH, cV, cD))
                current_img = cA
            coeffs.append(current_img)
            coeffs_list.append(coeffs)

        # 低频融合
        cA_list = [c[-1] for c in coeffs_list]
        window_size_low = max(3, window_size // 2)
        var_maps = []
        for cA in cA_list:
            local_mean = uniform_filter(cA, size=window_size_low, mode='mirror')
            local_sq_mean = uniform_filter(cA ** 2, size=window_size_low, mode='mirror')
            var = local_sq_mean - local_mean ** 2
            var_maps.append(var)
        var_stack = np.stack(var_maps, axis=-1)
        max_indices = np.argmax(var_stack, axis=-1)
        fused_cA = np.take_along_axis(np.stack(cA_list, axis=-1), max_indices[..., None], axis=-1).squeeze()
        fused_coeffs = [fused_cA]

        # 高频融合
        def local_fusion(stack, w_size=window_size):
            if w_size == 1:
                max_indices = np.argmax(np.abs(stack), axis=-1)
                return np.take_along_axis(stack, max_indices[..., None], axis=-1).squeeze()
            energy = np.zeros_like(stack, dtype=np.float32)
            for n in range(stack.shape[-1]):
                energy[..., n] = uniform_filter(np.abs(stack[..., n]), size=w_size, mode='mirror') * (w_size ** 2)
            max_indices = np.argmax(energy, axis=-1)
            return np.take_along_axis(stack, max_indices[..., None], axis=-1).squeeze()

        for l in range(level - 1, -1, -1):
            all_cH = [c[l][0] for c in coeffs_list]
            all_cV = [c[l][1] for c in coeffs_list]
            all_cD = [c[l][2] for c in coeffs_list]
            stack_cH = np.stack(all_cH, axis=-1)
            stack_cV = np.stack(all_cV, axis=-1)
            stack_cD = np.stack(all_cD, axis=-1)
            fused_cH = local_fusion(stack_cH, w_size=window_size)
            fused_cV = local_fusion(stack_cV, w_size=window_size)
            fused_cD = local_fusion(stack_cD, w_size=window_size)
            fused_coeffs.insert(0, (fused_cH, fused_cV, fused_cD))

        reconstructed = fused_coeffs[-1]
        for l in range(level):
            cH, cV, cD = fused_coeffs[len(fused_coeffs) - l - 2]
            reconstructed = pywt.idwt2((reconstructed, (cH, cV, cD)), wavelet, mode='periodization')
        reconstructed = reconstructed[:h_small, :w_small]

        # ★ 放大到原图分辨率
        if self._full_h and self._full_w:
            reconstructed = cv2.resize(reconstructed, (self._full_w, self._full_h),
                                       interpolation=cv2.INTER_LINEAR)

        self.focusing = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return self.focusing

    def AutoFocusing_WaveLet_GPU(self):
        h, w, num_images = self.stack.shape
        scale = self.rcf_scale
        h_small, w_small = h // scale, w // scale
        stack_small = np.stack([
            cv2.resize(self.stack[:, :, i], (w_small, h_small), interpolation=cv2.INTER_LINEAR)
            for i in range(num_images)
        ], axis=-1)

        wavelet = 'db2'
        window_size = 5
        level = 3

        stack_np = stack_small.astype(np.float32)
        stack_tensor = torch.from_numpy(stack_np).permute(2, 0, 1).unsqueeze(1).to(self.device)

        min_size = 2 ** (level + 1)
        pad_h = (min_size - (h_small % min_size)) % min_size
        pad_w = (min_size - (w_small % min_size)) % min_size
        padded = F.pad(stack_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        dwt = DWTForward(J=level, wave=wavelet, mode='periodization').to(self.device)
        yl, yh = dwt(padded)

        window_size_low = max(3, window_size // 2)
        pad_size = (window_size_low - 1) // 2
        padded_yl = F.pad(yl, (pad_size,) * 4, mode='reflect')
        local_mean = F.avg_pool2d(padded_yl, window_size_low, stride=1)
        local_sq_mean = F.avg_pool2d(padded_yl ** 2, window_size_low, stride=1)
        var = local_sq_mean - local_mean ** 2
        max_indices = torch.argmax(var, dim=0, keepdim=True)
        fused_cA = torch.gather(yl, 0, max_indices)
        fused_coeffs = [fused_cA]

        def fuse_high(coeff_stack, w_size):
            if w_size == 1:
                return coeff_stack.max(dim=0, keepdim=True)[0]
            pad_size = (w_size - 1) // 2
            padded = F.pad(torch.abs(coeff_stack), (pad_size,) * 4, mode='reflect')
            energy = F.avg_pool2d(padded, w_size, stride=1) * (w_size ** 2)
            max_indices = torch.argmax(energy, dim=0, keepdim=True)
            return torch.gather(coeff_stack, 0, max_indices)

        fused_yh = []
        for l in range(level):
            cH = yh[l][:, :, 0, :, :]
            cV = yh[l][:, :, 1, :, :]
            cD = yh[l][:, :, 2, :, :]
            fused_cH = fuse_high(cH, window_size)
            fused_cV = fuse_high(cV, window_size)
            fused_cD = fuse_high(cD, window_size)
            fused_high = torch.stack([fused_cH, fused_cV, fused_cD], dim=2)
            fused_yh.append(fused_high)

        idwt = DWTInverse(wave=wavelet, mode='periodization').to(self.device)
        reconstructed = idwt((fused_coeffs[-1], fused_yh))
        reconstructed = reconstructed[:, :, :h_small, :w_small]
        reconstructed = F.interpolate(reconstructed, size=(h_small, w_small), mode='bicubic', align_corners=False)
        result = reconstructed.squeeze().cpu().numpy()

        # ★ 放大到原图分辨率
        if self._full_h and self._full_w:
            result = cv2.resize(result, (self._full_w, self._full_h), interpolation=cv2.INTER_LINEAR)

        self.focusing = np.clip(result, 0, 255).astype(np.uint8)
        return self.focusing

    def _paste_particles_fullres(self, canvas, particles, positions):
        """将颗粒贴到原图分辨率画布上。canvas 为全分辨率，particles/positions 为 rcf_scale 缩放尺寸。"""
        ch, cw = canvas.shape[:2]
        s = self.rcf_scale
        for i in range(len(particles)):
            left, top, right, bottom = positions[i]
            if (bottom - top) * (right - left) <= self.min_particle_area:
                continue
            particle = particles[i]
            if particle.ndim == 3:
                particle = cv2.cvtColor(particle, cv2.COLOR_BGR2GRAY) if particle.shape[2] == 3 else particle[:, :, 0]
            particle = particle.astype(np.uint8)
            # 坐标和颗粒放大到原图分辨率
            l2, t2 = max(0, left * s), max(0, top * s)
            r2, b2 = min(cw, right * s), min(ch, bottom * s)
            if r2 <= l2 or b2 <= t2:
                continue
            particle_fit = cv2.resize(particle, (r2 - l2, b2 - t2), interpolation=cv2.INTER_LINEAR)
            canvas[t2:b2, l2:r2] = particle_fit

    def AutoFocusing_Wavelet_AI_CPU(self):
        wavelet_processed = self.AutoFocusing_WaveLet_CPU()
        particles, positions = self.AutoFocusing_AI_CPU(get_position=True)
        self._paste_particles_fullres(wavelet_processed, particles, positions)
        self.focusing = wavelet_processed

    def AutoFocusing_Wavelet_AI_GPU(self):
        wavelet_processed = self.AutoFocusing_WaveLet_GPU()
        particles, positions = self.AutoFocusing_AI_GPU(get_position=True)
        self._paste_particles_fullres(wavelet_processed, particles, positions)
        self.focusing = wavelet_processed

    def AutoFocusing_Gradient_Variance_CPU(self):
        h_full, w_full = self.stack.shape[0], self.stack.shape[1]
        num_images = self.stack.shape[2]

        scale = self.rcf_scale
        h_small, w_small = h_full // scale, w_full // scale
        images = [cv2.resize(self.stack[:, :, i], (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                  for i in range(num_images)]

        window_size = 5  # 与小波方法中的window_size保持一致

        # 计算局部方差
        var_maps = []
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)

        for img in images:
            img_float = img.astype(np.float32)
            # img_float = cv2.copyMakeBorder(img_float, window_size // 2, window_size // 2, window_size // 2,
            #                                 window_size // 2, cv2.BORDER_REFLECT)
            mean = cv2.filter2D(img_float, -1, kernel)
            sq_mean = cv2.filter2D(img_float ** 2, -1, kernel)
            variance = sq_mean - mean ** 2
            var_maps.append(variance)

        # 选择方差最大的图像进行融合
        var_stack = np.dstack(var_maps)
        max_indices = np.argmax(var_stack, axis=2)

        # 融合图像
        fused = np.zeros_like(images[0], dtype=np.float32)
        for n in range(len(images)):
            mask = (max_indices == n)
            fused[mask] = images[n][mask]

        # resize 回原图分辨率
        if self._full_h and self._full_w:
            fused = cv2.resize(fused, (self._full_w, self._full_h), interpolation=cv2.INTER_LINEAR)

        self.focusing = np.clip(fused, 0, 255).astype(np.uint8)
        return self.focusing

    # def AutoFocusing_Gradient_Variance_GPU(self):
    #     # 原始数据维度 [H, W, num_images]
    #     device = torch.device('cuda')
    #
    #     # 一次性将整个堆栈转移到GPU
    #     stack_tensor = torch.from_numpy(self.stack.astype(np.float32)).permute(2, 0, 1).to(device)  # [num_images, H, W]
    #     num_images, H, W = stack_tensor.shape
    #     window_size = 55
    #
    #     # 添加通道维度 [num_images, 1, H, W]
    #     stack_tensor = stack_tensor.unsqueeze(1)
    #
    #     # 创建卷积核 - 修正维度问题
    #     kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size ** 2)
    #
    #     # 计算均值和平方均值 (移除groups参数)
    #     mean = F.conv2d(stack_tensor, kernel, padding=window_size // 2)
    #     sq_mean = F.conv2d(stack_tensor ** 2, kernel, padding=window_size // 2)
    #
    #     # 计算方差 [num_images, 1, H, W]
    #     variance = sq_mean - mean ** 2
    #
    #     # 选择方差最大的图像索引 [H, W]
    #     max_indices = torch.argmax(variance.squeeze(1), dim=0)  # 沿图像维度取最大值
    #
    #     # 使用高级索引直接构建融合图像
    #     fused = torch.gather(
    #         input=stack_tensor.squeeze(1),  # [num_images, H, W]
    #         dim=0,
    #         index=max_indices.unsqueeze(0)  # [1, H, W]
    #     ).squeeze(0)
    #
    #     # 移回CPU并转换类型
    #     self.focusing = fused.cpu().numpy().astype(np.uint8)
    #     return self.focusing

    def AutoFocusing_Gradient_Variance_GPU(self):
        h_full, w_full = self.stack.shape[0], self.stack.shape[1]
        num_images = self.stack.shape[2]

        scale = self.rcf_scale
        h_small, w_small = h_full // scale, w_full // scale
        images = [cv2.resize(self.stack[:, :, i], (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                  for i in range(num_images)]

        window_size = 5

        device = torch.device('cuda')
        images_gpu = [torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) for img in images]

        kernel = torch.ones((1, 1, window_size, window_size), dtype=torch.float32).to(device) / (window_size ** 2)
        var_maps = []

        for img in images_gpu:
            mean = F.conv2d(img, kernel, padding=window_size // 2)
            sq_mean = F.conv2d(img ** 2, kernel, padding=window_size // 2)
            variance = sq_mean - mean ** 2
            var_maps.append(variance.squeeze(1))

        var_stack = torch.stack(var_maps, dim=-1)
        max_indices = torch.argmax(var_stack, dim=-1)

        fused = torch.zeros_like(images_gpu[0].squeeze(0).squeeze(0), dtype=torch.float32).to(device)
        for n in range(len(images_gpu)):
            mask = (max_indices == n).squeeze(0)
            fused[mask] = images_gpu[n].squeeze(0).squeeze(0)[mask]

        fused_cpu = fused.cpu().numpy()
        if self._full_h and self._full_w:
            fused_cpu = cv2.resize(fused_cpu, (self._full_w, self._full_h), interpolation=cv2.INTER_LINEAR)
        self.focusing = fused_cpu.astype(np.uint8)
        return self.focusing

    def AutoFocusing_Gradient_Variance_AI_CPU(self):
        wavelet_processed = self.AutoFocusing_Gradient_Variance_CPU()
        particles, positions = self.AutoFocusing_AI_CPU(get_position=True)
        self._paste_particles_fullres(wavelet_processed, particles, positions)
        self.focusing = wavelet_processed

    def AutoFocusing_Gradient_Variance_AI_GPU(self):
        wavelet_processed = self.AutoFocusing_Gradient_Variance_GPU()
        particles, positions = self.AutoFocusing_AI_GPU(get_position=True)
        self._paste_particles_fullres(wavelet_processed, particles, positions)
        self.focusing = wavelet_processed

    def AutoFocusing_AI_CPU(self, get_position=False):
        # 强制使用 CPU 设备
        device = torch.device('cpu')
        self.rcfModel.to(device)
        self.rcfModel.eval()

        # 设置 CPU 推理线程数
        if hasattr(self, 'cpu_num') and self.cpu_num:
            torch.set_num_threads(self.cpu_num)

        def create_square_mosaic(images, fill_value=128):
            """（保持不变）"""
            if not images:
                return None
            heights = [img.shape[0] for img in images]
            widths = [img.shape[1] for img in images]
            max_height = max(heights)
            max_width = max(widths)
            num_images = len(images)
            num_rows = int(np.sqrt(num_images))
            num_cols = (num_images + num_rows - 1) // num_rows
            mosaic_height = num_rows * max_height
            mosaic_width = num_cols * max_width
            mosaic = np.full((mosaic_height, mosaic_width), fill_value, dtype=np.uint8)
            for idx, img in enumerate(images):
                height, width = img.shape
                row = idx // num_cols
                col = idx % num_cols
                start_row = row * max_height
                start_col = col * max_width
                mosaic[start_row:start_row + height, start_col:start_col + width] = img
            return mosaic

        # 颗粒追踪
        mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
        tracker = Kalman_tracker(image_folder=None,
                                 output_folder=None,
                                 yolo_model=self.yoloModel,
                                 sort=mot_tracker,
                                 image_stack=self.stack,
                                 device=device)
        if get_position:
            stacks, stacks_ori, names, positions = tracker.get_stacks(get_position=get_position)
            self.focusing_xy = positions
        else:
            stacks, stacks_ori, names = tracker.get_stacks()

        processor = RCFProcessor(self.rcfModel, device, stacks, stacks_ori, rcf_scale=1,
                                 z_list=self._hologram.reconstruction_z)

        # ★ 使用线程池代替进程池，避免 GPU/CPU 跨设备错误
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.cpu_num if self.cpu_num else 4) as executor:
            results = list(executor.map(processor.process_stack, range(len(stacks))))

        focused_particles = []
        focusing_zs = []
        for focused_img, focusing_z in results:
            focused_particles.append(focused_img)
            focusing_zs.append(focusing_z)

        self.focusing_z = focusing_zs

        if get_position:
            return focused_particles, positions
        else:
            self.focusing = create_square_mosaic(focused_particles)
            for i in range(len(focused_particles)):
                self.focusing_each[str(i)] = focused_particles[i]

    def AutoFocusing_AI_GPU(self, get_position=False):
        # 颗粒子图拼接函数
        def create_square_mosaic(images, fill_value=128):
            """
            将灰度图像列表拼接成一张趋于方形的灰度图，保持原有子图尺寸，缺失部分用指定灰度值填充。

            参数:
            images (list of np.ndarray): 灰度图像列表
            fill_value (int): 填充空缺区域的灰度值，默认为128

            返回:
            np.ndarray: 拼接后的灰度图
            """
            if not images:
                return None

            # 获取所有子图的最大高度和宽度
            heights = [img.shape[0] for img in images]
            widths = [img.shape[1] for img in images]
            max_height = max(heights)
            max_width = max(widths)

            # 计算行数和列数，使得排列趋于方形
            num_images = len(images)
            num_rows = int(np.sqrt(num_images))
            num_cols = (num_images + num_rows - 1) // num_rows  # 向上取整

            # 创建空白画布
            mosaic_height = num_rows * max_height
            mosaic_width = num_cols * max_width
            mosaic = np.full((mosaic_height, mosaic_width), fill_value, dtype=np.uint8)

            # 将子图逐个放置到画布上
            for idx, img in enumerate(images):
                # 获取子图的高度和宽度
                height, width = img.shape

                # 计算放置位置
                row = idx // num_cols
                col = idx % num_cols

                # 计算子图在画布上的起始行和列
                start_row = row * max_height
                start_col = col * max_width

                # 将子图放置到画布上
                mosaic[start_row:start_row + height, start_col:start_col + width] = img

            return mosaic

        focused_particles = []
        c = time.time()
        # 颗粒追踪，get_tracks函数耗时3s
        mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
        tracker = Kalman_tracker(image_folder=None, output_folder=None,
                                 yolo_model=self.yoloModel, sort=mot_tracker, image_stack=self.stack)

        if get_position:
            stacks, stacks_ori, names, positions = tracker.get_stacks(get_position=get_position)    # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)
            self.focusing_xy = positions    # 列表[[x1,y1,x2,y2]
        else:
            stacks, stacks_ori, names = tracker.get_stacks()
        d = time.time()

        e = time.time()
        for i, stack in enumerate(stacks):
            stack_ori = stacks_ori[i]
            # V0
            # img_tmp, name_tmp = rcf_predict(self.rcfModel, get_dataset(stack, stack_ori, k_size=self.rcf_scale), device=self.device)

            # V1
            dataset = RCFLoader(stack, stack_ori, k_size=1)
            loader = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True)

            img_tmp, index_tmp = rcf_predict_V1(self.rcfModel, loader, device=self.device)
            img_tmp = torch.squeeze(img_tmp.squeeze()).cpu().numpy()
            focused_particles.append(img_tmp)  # 聚焦颗粒图像列表, note

            self.focusing_z.append(self._hologram.reconstruction_z[index_tmp])      # 各个颗粒的聚焦z

            # 保存显示
            # cv2.imwrite(os.path.join(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\ai_output', "{}.jpg".format(i)), img_tmp)
        f = time.time()
        if get_position:
            return focused_particles, positions
        else:
            # 拼接图
            self.focusing = create_square_mosaic(focused_particles)
            # 单个颗粒
            for i in range(len(focused_particles)):
                self.focusing_each[str(i)] = focused_particles[i]

    def AutoFocusing_AI_GPU_V1(self, get_position: bool = False):
        """一次性完成所有颗粒的自动聚焦，输出每个颗粒的最清晰图像
        参数
        ----
        get_position : bool
            若为 True，则同时返回各颗粒在原始帧中的坐标。
        返回
        ----
        focused_particles : List[np.ndarray]
            每颗粒 1 张聚焦图（与原 img 相同尺寸）。
        positions : List[Tuple[int, int, int, int]]  (仅当 get_position 为 True)
            Sort/Kalman 给出的 (x1, y1, x2, y2) 位置。
        """

        # --------------------------- 辅助函数 ---------------------------
        def create_square_mosaic(images, fill_value=128):
            """把若干单通道图拼成接近正方形的马赛克图，调试/展示用"""
            if not images:
                return None
            heights = [img.shape[0] for img in images]
            widths = [img.shape[1] for img in images]
            h_max, w_max = max(heights), max(widths)
            n = len(images)
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            mosaic = np.full((rows * h_max, cols * w_max), fill_value, np.uint8)

            for idx, img in enumerate(images):
                r, c = divmod(idx, cols)
                h, w = img.shape
                mosaic[r * h_max: r * h_max + h, c * w_max: c * w_max + w] = img
            return mosaic

        # --------------------- 颗粒追踪，生成 stack ----------------------
        mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
        tracker = Kalman_tracker(
            image_folder=None,
            output_folder=None,
            yolo_model=self.yoloModel,
            sort=mot_tracker,
            image_stack=self.stack
        )

        if get_position:
            stacks, stacks_ori, names, positions = tracker.get_stacks(get_position=True)
        else:
            stacks, stacks_ori, names = tracker.get_stacks()

        # ---------------------- 构建数据集列表 --------------------------
        all_datasets, particle_sizes = [], []
        H_max, W_max = 0, 0
        for stack, stack_ori in zip(stacks, stacks_ori):
            # stack[0] 就代表该颗粒内所有 img 的统一尺寸
            h_img, w_img = stack[0].shape  # img 尺寸
            h_ori, w_ori = stack_ori[0].shape  # img_ori 尺寸

            H_max = max(H_max, h_img, h_ori)
            W_max = max(W_max, w_img, w_ori)

            ds = RCFLoader(stack, stack_ori, k_size=1)
            all_datasets.append(ds)
            particle_sizes.append(len(ds))  # 一般是 26

        combined_dataset = ConcatDataset(all_datasets)

        # ----------- 2. 自定义 collate_fn，实现零填充 + 原尺寸记录 --------
        # ============================================
        # 适用 img:(3,H,W) ；img_ori:(H,W) 的 pad_collate
        # ============================================
        def pad_collate(batch):
            """
            返回：
                imgs_padded     (N, 3, H_max, W_max)
                img_oris_padded (N, 1, H_max, W_max)
                orig_sizes      [(H_img, W_img, H_ori, W_ori), ...]
            """
            imgs, img_oris, orig_sizes = [], [], []

            for img, img_ori in batch:
                # ---------------- img (3CH) ----------------
                _, h_i, w_i = img.shape
                pad_h_i, pad_w_i = H_max - h_i, W_max - w_i
                img_padded = F.pad(img, (0, pad_w_i, 0, pad_h_i), value=0)

                # ---------------- img_ori (1CH) -------------
                if img_ori.ndim == 2:  # 先扩一维到 1×H×W
                    img_ori = img_ori.unsqueeze(0)
                _, h_o, w_o = img_ori.shape
                pad_h_o, pad_w_o = H_max - h_o, W_max - w_o
                img_ori_padded = F.pad(img_ori, (0, pad_w_o, 0, pad_h_o), value=0)
                img_ori_padded = img_ori_padded.squeeze()

                # ---------------- collect -------------------
                imgs.append(img_padded)
                img_oris.append(img_ori_padded)
                orig_sizes.append((h_i, w_i, h_o, w_o))

            return torch.stack(imgs), torch.stack(img_oris), orig_sizes

        combined_loader = DataLoader(
            combined_dataset,
            batch_size=256,
            num_workers=0,
            pin_memory=True,
            collate_fn=pad_collate
        )

        # ------------------------ 3. 模型推理 --------------------------
        self.rcfModel.eval()
        focused_tensors = rcf_predict_V2(  # ← 你已有的推理函数
            self.rcfModel,
            combined_loader,
            particle_sizes,  # 用于 rcf_predict_V2 拆分
            device=self.device
        )  # 返回 torch.Tensor 列表，每张大小 (1, H_max, W_max)

        # -------------------- 4. 剪裁回原始尺寸 ------------------------
        focused_particles = []
        idx = 0
        for size in particle_sizes:  # 按颗粒拆分
            best_img = focused_tensors[idx]  # 每颗粒只保留 1 张最聚焦
            idx += 1
            H_pad, W_pad = best_img.shape
            # 裁掉填充。因为每颗粒内部 img 尺寸相同，用第一张记录的尺寸即可
            h0, w0 = stacks[0][0].shape  # 原尺寸（同颗粒同尺寸）
            if isinstance(best_img, torch.Tensor):  # 兼容老版本返回 tensor 的情况
                best_img = best_img.squeeze().cpu().numpy()
                # 若已是 numpy，则什么都不用做

            focused_particles.append(best_img)

        # ------------------- 5. 结果组织 & 返回 ------------------------
        if get_position:
            return focused_particles, positions
        else:
            self.focusing = create_square_mosaic(focused_particles)
            for i, img in enumerate(focused_particles):
                self.focusing_each[str(i)] = img


    # ========================================================================
    # PCHIP 快速自聚焦方法
    # ========================================================================
    def AutoFocusing_PCHIP(self):
        """
        PCHIP 快速自聚焦（CPU 版）

        假设所有颗粒位于同一聚焦面，利用 PCHIP 插值代理模型
        搜索最优聚焦深度 z，仅需 ~30 次单截面重建即可找到全局最优。

        聚焦分数 = RCF 边缘预测 → 亮度集中度（非零像素方差）
        """
        fast_focus = FastFocusPCHIP(
            hologram=self._hologram,
            config=self._config,
            rcf_model=(self.rcfModel if hasattr(self, 'rcfModel') else None),
            device=self.device,
            k_size=self.rcf_scale
        )
        image, optimal_z = fast_focus.run()

        self.focusing = image
        self.focusing_z = [optimal_z]
        self.focusing_each = {}

        return image

    def AutoFocusing_PCHIP_GPU(self):
        """
        PCHIP 快速自聚焦（GPU 版，使用 CuPy 加速角谱法重建）
        RCF 推理仍使用 PyTorch GPU
        """
        try:
            fast_focus = FastFocusPCHIP_GPU(
                hologram=self._hologram,
                config=self._config,
                rcf_model=(self.rcfModel if hasattr(self, 'rcfModel') else None),
                device=self.device,
                k_size=self.rcf_scale
            )
            image, optimal_z = fast_focus.run()

            self.focusing = image
            self.focusing_z = [optimal_z]
            self.focusing_each = {}

            return image

        except ImportError:
            print("[Warning] CuPy not available, falling back to CPU PCHIP focusing.")
            return self.AutoFocusing_PCHIP()
    # ========================================================================
    # Multi_PCHIP 多颗粒 PCHIP 自聚焦
    # ========================================================================
    def AutoFocusing_Multi_PCHIP(self):
        focuser = MultiFocusPCHIP(self._hologram, self._config)
        focuser.run()
        self.focusing = focuser.focusing
        self.focusing_z = focuser.focusing_z
        self.focusing_xy = focuser.focusing_xy
        self.focusing_each = focuser.focusing_each
        return self.focusing

    def AutoFocusing_Multi_PCHIP_GPU(self):
        try:
            focuser = MultiFocusPCHIP_GPU(self._hologram, self._config)
            focuser.run()
            self.focusing = focuser.focusing
            self.focusing_z = focuser.focusing_z
            self.focusing_xy = focuser.focusing_xy
            self.focusing_each = focuser.focusing_each
            return self.focusing
        except ImportError:
            print("[Warning] CuPy not available, falling back to CPU Multi_PCHIP.")
            return self.AutoFocusing_Multi_PCHIP()

def batch_Focusing(root=r'F:\lichenghao\data', method='AI_GPU'):
    # 指定使用的GPU
    import os
    gpu_id = 5
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 设置设备为指定GPU
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # print(f"Using device: {device} (GPU {gpu_id})")
    CUDA_Available = (device.type == 'cuda')
    # print(CUDA_Available)

    yolo_pth = r'F:\dongjiayao\Pycharm\HoloLab\models\yolo_detection.pth'

    yolo_model = YOLO(input_shape=[640, 640],
                      phi='s',
                      model_path=yolo_pth,
                      cuda=CUDA_Available,
                      letterbox_image=True,
                      confidence=0.5,
                      nms_iou=0.3,
                      device=device)

    sub_roots = os.listdir(root)

    for sub_root in sub_roots:
        save_pth = os.path.join(root, sub_root, 'single_DJY_ASPP')
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        hologram = MockData(data_path=os.path.join(root, sub_root))
        reconstruction_stack = hologram.holo.reconstruction  # 形状为(4096, 4508, num)

        # yolo_model外部传入，避免重复加载
        Focus = Focusing(reconstruction_stack, gpu_id=gpu_id, yoloModel=yolo_model, device=device)

        # start0 = time.time()
        gpu_result = Focus.run(method=method)

        idx = 0
        for img in gpu_result:
            idx += 1
            cv2.imwrite(os.path.join(save_pth, "{}.png".format(idx)), img)

        # cv2.imwrite(os.path.join(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\output', "{}.png".format(idx)), gpu_result)

        # end0 = time.time()
        # print("GPU耗时：{:.2f}".format(end0 - start0))

def batch_Focusing_with_bar(config):
    # F:\lichenghao\data\2025.0429,9.2xSSC,PSDtry\2-nachangshi\chuli\5\chuli
    # F:\lichenghao\data\2025.0429,9.2xSSC,PSDtry\2-nachangshi\chuli\5again\150
    # F:\lichenghao\data\2025.0429,9.2xSSC,PSDtry\2-nachangshi\chuli\5\tmp
    from tqdm import tqdm

    # 指定使用的GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.focusing['gpu_id'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    # 设置设备为指定GPU
    device = torch.device(config.focusing['device'])
    # print(f"Using device: {device} (GPU {config.focusing['gpu_id']})")
    CUDA_Available = (device.type == 'cuda')

    # if True:
    #     # 强制设置设备为CPU
    #     device = torch.device('cpu')
    #     print(f"Using device: {device}")
    #     torch.backends.cudnn.enabled = False  # 禁用cuDNN

    root = config.focusing['batch_root']

    # 模型定义和加载
    yolo_model = YOLO(input_shape=[640, 640],
                      phi='s',
                      model_path=config.focusing['yolo_model_path'],
                      cuda=CUDA_Available,
                      letterbox_image=True,
                      confidence=0.5,
                      nms_iou=0.3,
                      device=device)
    rcf_model = RCF(device)
    rcf_model.to(device)
    checkpoint = torch.load(config.focusing['rcf_model_path'])
    rcf_model.load_state_dict(checkpoint['state_dict'])

    sub_roots = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    # 新增：创建进度条
    progress_bar = tqdm(
        sub_roots,
        desc="Processing directories",
        unit="dir",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for sub_root in progress_bar:  # 修改循环为使用进度条
        save_pth = os.path.join(root, sub_root, 'dof_result')
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        progress_bar.set_postfix_str(f"Processing {sub_root}")

        # 定义数据和方法
        hologram = Hologram(config=HoloConfig())
        Focus = Focusing(hologram, config=config, yoloModel=yolo_model, rcfModel=rcf_model)

        # 根据文件路径读取保存的重建图
        data_path = os.path.join(root, sub_root, 'reconstruction')
        recons_files = os.listdir(data_path)
        reconstruction = {}
        idx = 0
        for recons_file in recons_files:
            idx += 1
            recons = cv2.imread(os.path.join(data_path, recons_file), 0)
            reconstruction[idx] = recons

        # 不从hologram类中读取，而是在这里外部赋值
        stack_list = [reconstruction[i] for i in reconstruction.keys()]
        # 堆叠为三维数组 (height, width, num)
        Focus.stack = np.stack(stack_list, axis=2).astype(np.float64)  # 或 axis=-1

        # 运行
        gpu_result = Focus.run()

        # TODO 如果返回单张图像
        cv2.imwrite(os.path.join(save_pth, f"GPU.png"), gpu_result)

    # 处理完成提示
    progress_bar.close()
    print("\nAll directories processed successfully!")

if __name__ == '__main__':
    print('Utils Focusing Module', end='\n\n')

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