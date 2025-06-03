import sys
from concurrent.futures import ProcessPoolExecutor  # CPU并行化
from pathlib import Path

from module.hologram import Hologram
from utils.utils_for_focusing.rcf.data_loader import prepare_image_PIL, convert_to_rgb

# 将项目根目录添加到 sys.path
root_dir = Path(__file__).parent.parent  # 根据实际情况调整层级
# print('Check the Root:', root_dir)
sys.path.append(str(root_dir))

import os
import time
from utils.mock_data import MockData
import numpy as np
import pywt  # pip install PyWavelets
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader

# HoloTrack
from utils.utils_for_focusing.final import RCFLoader, rcf_predict

# GPU加速所需库
try:
    import torch
    import torch.nn.functional as F
    from pytorch_wavelets import DWTForward, DWTInverse
    # print("Using CUDA:", print(torch.cuda.is_available()))
except Exception as e:
    print(f"[Module] {e}")

from utils.utils_for_focusing.yoloV8.yolo import YOLO
from utils.utils_for_focusing.sort.kalman_trace_iou_center import Kalman_tracker, Sort
from utils.utils_for_focusing.rcf.models.RCF import RCF
from utils.utils_for_focusing.rcf.models.RCF_ASPP import RCF_ASPP

import cv2
from module.config import HoloConfig


'''
1、输入：全息图 中的 重建图像簇（重建完成之后的）self.reconstruction
2、输出：聚焦完成的图 self.focusing；其中，AI方法

- 上述这些参数，在Hologram这个类中已经定义，可以直接返回给它的实体
- 先看看Hologram这个类中有哪些量

'''

class RCFProcessor:
    def __init__(self, rcf_model, device, stacks, stacks_ori, rcf_scale):
        self.rcf_model = rcf_model
        self.device = device
        self.stacks = stacks
        self.stacks_ori = stacks_ori
        self.rcf_scale = rcf_scale

    def process_stack(self, stack_idx):
        stack = self.stacks[stack_idx]
        stack_ori = self.stacks_ori[stack_idx]
        # dataset = RCFLoader(stack, stack_ori, k_size=8)
        # loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=False)
        # img_tmp, name_tmp = rcf_predict(self.rcf_model, loader, device=self.device)

        img_tmp, name_tmp = rcf_predict(self.rcf_model, get_dataset(stack, stack_ori, k_size=self.rcf_scale),
                                        device=self.device)

        return torch.squeeze(img_tmp.squeeze()).cpu().numpy()

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
    def __init__(self, hologram, config, **kwargs):
        self.stack_dict = hologram.reconstruction
        stack_list = [self.stack_dict[i] for i in self.stack_dict.keys()]
        # 堆叠为三维数组 (height, width, num)
        self.stack = np.stack(stack_list, axis=2).astype(np.float64) * 255  # 或 axis=-1

        # self.stack = stack  # 重建图簇，np.array(float32)，位于CPU
        self.gpu_id = config.focusing['gpu_id']
        self.cpu_num = config.focusing['cpu_num']     # cpu线程数
        self.yoloModel_path = config.focusing['yolo_model_path']
        self.rcfModel_path = config.focusing['rcf_model_path']
        self.device = torch.device(config.focusing['device'])
        self.get_model = config.focusing['get_model']   # 如果为True，将直接给Focus类传入模型本身，而不是根据路径加载模型
        self.method = config.focusing['method']
        self.rcf_scale = config.focusing['rcf_scale']       # 用于rcf图像缩放

        # 检查**kwargs中是否传入了yoloModel参数
        if 'yoloModel' in kwargs:
            self.yoloModel = kwargs['yoloModel']
            self.rcfModel = kwargs['rcfModel']

        elif self.method in ['AI', 'AI_Wavelet', 'AI_Gradient']:
            # 从路径加载模型
            self.load_yolo_model()

    # 执行函数，需要输入方法名称
    def run(self):
        result = None

        if self.method == 'Wavelet' and self.device.type == 'cpu':
            result = self.AutoFocusing_WaveLet_CPU()

        elif self.method == 'Gradient' and self.device.type == 'cpu':
            result = self.AutoFocusing_Gradient_Variance_CPU()

        elif self.method == 'AI' and self.device.type == 'cpu':
            result = self.AutoFocusing_Machine_Learning_CPU()

        elif self.method == 'Wavelet' and self.device.type == 'cuda':
            result = self.AutoFocusing_WaveLet_GPU()

        elif self.method == 'Gradient' and self.device.type == 'cuda':
            result = self.AutoFocusing_Gradient_Variance_GPU()

        elif self.method == 'AI' and self.device.type == 'cuda':
            result = self.AutoFocusing_Machine_Learning_GPU()

        elif self.method == 'AI_Wavelet' and self.device.type == 'cpu':
            result = self.AutoFocusing_AI_Wavelet_CPU()

        elif self.method == 'AI_Gradient' and self.device.type == 'cpu':
            result = self.AutoFocusing_AI_Gradient_CPU()

        elif self.method == 'AI_Wavelet' and self.device.type == 'cuda':
            result = self.AutoFocusing_AI_Wavelet_GPU()

        elif self.method == 'AI_Gradient' and self.device.type == 'cuda':
            result = self.AutoFocusing_AI_Gradient_GPU()

        else:
            print('No such choice!')

        return result

    # 在没有外部传入模型时，根据模型路径加载模型
    def load_yolo_model(self):
        if self.get_model:
            pass
        else:
            # 指定使用的GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            print(f"Using device: {self.device} (GPU {self.gpu_id})")
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

    def AutoFocusing_WaveLet_CPU(self):
        h, w, num_images = self.stack.shape

        # 以下为小波拓展所需参数
        wavelet = 'db2'
        window_size = 55
        fusion_mode = 'max'
        level = 3

        coeffs_list = []

        # 遍历并提取每个图像
        for i in range(self.stack.shape[2]):
            img = self.stack[:, :, i]
            current_img = img.copy()

            # 图像预处理，确保尺寸合理
            min_size = 2 ** (level + 1)
            base_h = min(s for s in (h, w))
            base_w = max(s for s in (h, w))

            base_h = (base_h // min_size) * min_size
            base_w = (base_w // min_size) * min_size

            pad_h = (min_size - (base_h % min_size)) % min_size
            pad_w = (min_size - (base_w % min_size)) % min_size

            original_size, padding = (base_h, base_w), (pad_h, pad_w)

            cropped = current_img[:base_h, :base_w]
            padded = np.pad(cropped, ((0, pad_h), (0, pad_w)), mode='symmetric').astype(np.float32)
            current_img = padded

            # 开始处理
            coeffs = []
            for _ in range(level):
                cA, (cH, cV, cD) = pywt.dwt2(current_img, wavelet, mode='periodization')
                coeffs.append((cH, cV, cD))
                current_img = cA
            coeffs.append(current_img)
            coeffs_list.append(coeffs)

        # 改进低频融合：选择局部方差最大的低频系数
        cA_list = [c[-1] for c in coeffs_list]
        window_size_low = max(3, window_size // 2)  # 低频窗口尺寸适当减小
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

        # 高频系数融合修正
        def local_fusion(stack, mode=fusion_mode, w_size=window_size):
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

        # 逆变换重建
        reconstructed = fused_coeffs[-1]
        for l in range(level):
            cH, cV, cD = fused_coeffs[len(fused_coeffs) - l - 2]
            reconstructed = pywt.idwt2((reconstructed, (cH, cV, cD)), wavelet, mode='periodization')

        target_h, target_w = original_size[0] + padding[0], original_size[1] + padding[1]
        reconstructed = reconstructed[:target_h, :target_w]

        # 回归原有尺寸
        reconstructed = cv2.resize(reconstructed, (w, h))

        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def AutoFocusing_Gradient_Variance_CPU(self):
        images = [self.stack[:, :, i] for i in range(self.stack.shape[2])]
        window_size = 55  # 与小波方法中的window_size保持一致

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

        return np.clip(fused, 0, 255).astype(np.uint8)

    def AutoFocusing_Machine_Learning_CPU(self, get_position=False):
        # 设置CPU并行计算
        torch.set_num_threads(self.cpu_num)      # todo 改掉，用多进程而不是多线程
        torch.backends.openmp.enabled = True
        torch.backends.mkldnn.enabled = True  # 启用MKL-DNN加速

        # 颗粒子图拼接函数
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

        focused_particles = []

        # 颗粒追踪, 耗费19s
        mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
        tracker = Kalman_tracker(image_folder=None,
                                 output_folder=None,
                                 yolo_model=self.yoloModel,
                                 sort=mot_tracker,
                                 image_stack=self.stack,
                                 device=self.device)

        if get_position:
            stacks, stacks_ori, names, positions = tracker.get_stacks(
                get_position=get_position)  # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)
        else:
            stacks, stacks_ori, names = tracker.get_stacks()

        # 并行化 RCF 模型的预测
        # 创建 RCFProcessor 实例
        processor = RCFProcessor(self.rcfModel, self.device, stacks, stacks_ori, rcf_scale=self.rcf_scale)

        # 使用 ProcessPoolExecutor.map 并行处理
        with ProcessPoolExecutor(max_workers=self.cpu_num) as executor:
            focused_particles = list(executor.map(processor.process_stack, range(len(stacks))))

        mosaic = create_square_mosaic(focused_particles)

        if get_position:
            return focused_particles, positions
        else:
            return mosaic

    def AutoFocusing_WaveLet_GPU(self):
        # print(torch.cuda.is_available())  # 应输出True才能使用GPU
        # print(torch.cuda.device_count())  # 可用GPU数量

        h, w, num_images = self.stack.shape
        wavelet = 'db2'
        window_size = 55
        level = 3

        # 将数据转移到GPU
        stack_np = self.stack.astype(np.float32)
        stack_tensor = torch.from_numpy(stack_np).permute(2, 0, 1).unsqueeze(1).to(self.device)  # (num_images, 1, h, w)

        # 预处理填充
        min_size = 2 ** (level + 1)
        base_h = min(h, w)
        base_w = max(h, w)
        base_h = (base_h // min_size) * min_size
        base_w = (base_w // min_size) * min_size
        pad_h = (min_size - (base_h % min_size)) % min_size
        pad_w = (min_size - (base_w % min_size)) % min_size

        # 裁剪并对称填充
        cropped = stack_tensor[:, :, :base_h, :base_w]
        padded = F.pad(cropped, (0, pad_w, 0, pad_h), mode='reflect')

        # 小波分解
        dwt = DWTForward(J=level, wave=wavelet, mode='periodization').to(self.device)
        yl, yh = dwt(padded)

        # 低频融合
        window_size_low = max(3, window_size // 2)
        pad_size = (window_size_low - 1) // 2

        # 计算局部方差
        padded_yl = F.pad(yl, (pad_size,) * 4, mode='reflect')
        local_mean = F.avg_pool2d(padded_yl, window_size_low, stride=1)
        local_sq_mean = F.avg_pool2d(padded_yl ** 2, window_size_low, stride=1)
        var = local_sq_mean - local_mean ** 2

        # 选择最大方差索引
        max_indices = torch.argmax(var, dim=0, keepdim=True)
        fused_cA = torch.gather(yl, 0, max_indices)
        fused_coeffs = [fused_cA]

        # 高频融合函数
        def fuse_high(coeff_stack, w_size):
            if w_size == 1:
                return coeff_stack.max(dim=0, keepdim=True)[0]
            pad_size = (w_size - 1) // 2
            padded = F.pad(torch.abs(coeff_stack), (pad_size,) * 4, mode='reflect')
            energy = F.avg_pool2d(padded, w_size, stride=1) * (w_size ** 2)
            max_indices = torch.argmax(energy, dim=0, keepdim=True)
            return torch.gather(coeff_stack, 0, max_indices)

        # 逐层融合高频系数
        fused_yh = []
        for l in range(level):
            current_yh = yh[l]  # 形状为 (num_images, 1, 3, h, w)
            cH = current_yh[:, :, 0, :, :]
            cV = current_yh[:, :, 1, :, :]
            cD = current_yh[:, :, 2, :, :]

            fused_cH = fuse_high(cH, window_size)
            fused_cV = fuse_high(cV, window_size)
            fused_cD = fuse_high(cD, window_size)
            # 合并三个高频分量到第三维度
            fused_high = torch.stack([fused_cH, fused_cV, fused_cD], dim=2)
            fused_yh.append(fused_high)

        # 小波逆变换
        idwt = DWTInverse(wave=wavelet, mode='periodization').to(self.device)
        reconstructed = idwt((fused_coeffs[-1], fused_yh))

        # 后处理
        result = reconstructed[:, :, :base_h + pad_h, :base_w + pad_w]
        result = F.interpolate(result, size=(h, w), mode='bicubic', align_corners=False)
        result = result.squeeze().cpu().numpy()
        result = cv2.resize(result, (w, h))
        return np.clip(result, 0, 255).astype(np.uint8)

    def AutoFocusing_Gradient_Variance_GPU(self):
        images = [self.stack[:, :, i] for i in range(self.stack.shape[2])]
        window_size = 55  # 与小波方法中的window_size保持一致

        # 将数据转移到GPU
        device = torch.device('cuda')
        images_gpu = [torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) for img in images]

        # 计算局部方差
        kernel = torch.ones((1, 1, window_size, window_size), dtype=torch.float32).to(device) / (window_size ** 2)
        var_maps = []

        for img in images_gpu:
            mean = F.conv2d(img, kernel, padding=window_size // 2)
            sq_mean = F.conv2d(img ** 2, kernel, padding=window_size // 2)
            variance = sq_mean - mean ** 2
            var_maps.append(variance.squeeze(1))  # 移除通道维度

        # 选择方差最大的图像进行融合
        var_stack = torch.stack(var_maps, dim=-1)  # (batch, h, w, num_images)
        max_indices = torch.argmax(var_stack, dim=-1)  # (batch, h, w)

        # 融合图像
        fused = torch.zeros_like(images_gpu[0].squeeze(0).squeeze(0), dtype=torch.float32).to(device)
        for n in range(len(images_gpu)):
            mask = (max_indices == n).squeeze(0)
            fused[mask] = images_gpu[n].squeeze(0).squeeze(0)[mask]

        # 将结果从GPU转移到CPU
        fused_cpu = fused.cpu().numpy()

        return np.clip(fused_cpu, 0, 255).astype(np.uint8)

    def AutoFocusing_Machine_Learning_GPU(self, get_position=False):
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

        # 颗粒追踪，get_tracks函数耗时3s
        mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
        tracker = Kalman_tracker(image_folder=None,
                                 output_folder=None,
                                 yolo_model=self.yoloModel,
                                 sort=mot_tracker,
                                 image_stack=self.stack)

        if get_position:
            stacks, stacks_ori, names, positions = tracker.get_stacks(get_position=get_position)    # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)
        else:
            stacks, stacks_ori, names = tracker.get_stacks()

        # 显示图像
        # stacks_ori[1][0].show()

        for i, stack in enumerate(stacks):
            stack_ori = stacks_ori[i]
            # dataset = RCFLoader(stack, stack_ori, k_size=6)
            # loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=False)
            # img_tmp, name_tmp = rcf_predict(rcf_model, loader)
            img_tmp, name_tmp = rcf_predict(self.rcfModel, get_dataset(stack, stack_ori, k_size=self.rcf_scale), device=self.device)

            img_tmp = torch.squeeze(img_tmp.squeeze()).cpu().numpy()
            focused_particles.append(img_tmp)  # 聚焦颗粒图像列表, note

            # 保存显示
            # cv2.imwrite(os.path.join(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\ai_output', "{}.jpg".format(i)), img_tmp)

        # 颗粒子图拼接
        mosaic = create_square_mosaic(focused_particles)

        if get_position:
            return focused_particles, positions
        else:
            return mosaic

    def AutoFocusing_AI_Wavelet_CPU(self):
        wavelet_processed = self.AutoFocusing_WaveLet_CPU()
        # wavelet_processed = self.AutoFocusing_Gradient_Variance_GPU()
        particles, positions = self.AutoFocusing_Machine_Learning_CPU(get_position=True)    # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)

        for i in range(len(particles)):
            # 提取位置信息
            left, top, right, bottom = positions[i]
            if (bottom - top) * (right - left) > 1600:
                # 提取对应的颗粒图像
                particle = particles[i].astype(np.uint8)

                wavelet_processed[top:bottom, left:right] = particle

        return wavelet_processed

    def AutoFocusing_AI_Gradient_CPU(self):
        # wavelet_processed = self.AutoFocusing_WaveLet_GPU()
        wavelet_processed = self.AutoFocusing_Gradient_Variance_CPU()
        particles, positions = self.AutoFocusing_Machine_Learning_CPU(get_position=True)    # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)

        for i in range(len(particles)):
            # 提取位置信息
            left, top, right, bottom = positions[i]
            if (bottom - top) * (right - left) > 1600:
                # 提取对应的颗粒图像
                particle = particles[i].astype(np.uint8)

                wavelet_processed[top:bottom, left:right] = particle

        return wavelet_processed

    def AutoFocusing_AI_Wavelet_GPU(self):
        wavelet_processed = self.AutoFocusing_WaveLet_GPU()
        # wavelet_processed = self.AutoFocusing_Gradient_Variance_GPU()
        particles, positions = self.AutoFocusing_Machine_Learning_GPU(get_position=True)    # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)

        for i in range(len(particles)):
            # 提取位置信息
            left, top, right, bottom = positions[i]
            if (bottom - top) * (right - left) > 1600:
                # 提取对应的颗粒图像
                particle = particles[i].astype(np.uint8)

                wavelet_processed[top:bottom, left:right] = particle

        return wavelet_processed

    def AutoFocusing_AI_Gradient_GPU(self):
        # wavelet_processed = self.AutoFocusing_WaveLet_GPU()
        wavelet_processed = self.AutoFocusing_Gradient_Variance_GPU()
        particles, positions = self.AutoFocusing_Machine_Learning_GPU(get_position=True)    # positions为列表，储存元组(left_ori, top_ori, right_ori, bottom_ori)

        for i in range(len(particles)):
            # 提取位置信息
            left, top, right, bottom = positions[i]
            if (bottom - top) * (right - left) > 1600:
                # 提取对应的颗粒图像
                particle = particles[i].astype(np.uint8)

                wavelet_processed[top:bottom, left:right] = particle

        return wavelet_processed

def batch_Focusing(root=r'F:\lichenghao\data\2025.0429,9.2xSSC,PSDtry\2-nachangshi\chuli\5\chuli',
                   method='AI_GPU'):
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
    # print('Utils Focusing Module', end='\n\n')

    '''
        预定义一个全息，并加载模拟数据，用于测试，包括
        - 原始全息图
        - 预处理后的图
        - 重建完成的图
        - 聚焦完成的图
        - 波前和相位
    '''
    # hologram = MockData(data_path=r'F:\lichenghao\data\2025.0429,9.2xSSC,PSDtry\2-nachangshi\chuli\5again\test\Basler_acA2440-75umPOL__24968847__20250429_195631606_1.bmp')
    # reconstruction_stack = hologram.holo.reconstruction  # 形状为(4096, 4508, num)

    # hologram = Hologram(config=HoloConfig())
    # Focus = Focusing(hologram, config=HoloConfig())

    # start0 = time.time()
    # cpu_result = Focus.run(method='AI_CPU')
    # cv2.imwrite(os.path.join(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\output', "CPU.png"), cpu_result)
    # end0 = time.time()
    # print("CPU耗时：{:.2f}".format(end0 - start0))

    # start0 = time.time()
    # gpu_result = Focus.run()     # TODO 选择自聚焦方法：Wavelet_CPU、Gradient_CPU、AI_CPU、 Wavelet_GPU、Gradient_GPU、AI_GPU
    # cv2.imwrite(os.path.join(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\output', "GPU.png"), gpu_result)
    # end0 = time.time()
    # print("GPU耗时：{:.2f}".format(end0 - start0))

    # 以下为批量处理代码，暂时不集成到UI中
    # batch_Focusing(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\tmp')
    # batch_Focusing()

    # 带进度条
    # batch_Focusing_with_bar(config=HoloConfig())
