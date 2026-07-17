"""
PCHIP-Based Fast Single-Plane Auto Focusing Module

单截面快速自聚焦模块：
- 假设所有颗粒位于同一聚焦面（不考虑三维分布）
- 使用 PCHIP 插值构建代理模型，通过贝叶斯优化策略快速搜索最优聚焦深度
- 相比传统多截面重建+聚焦方法，大幅减少重建次数

基本原理：
1. 在搜索区间内采样若干 z 值，对每个 z 做一次重建
2. 用重建图像的清晰度（方差）作为评价指标
3. 用 PCHIP 插值构建"z → 清晰度"的代理模型
4. 在代理模型上搜索最优 z
"""
import torch
from PIL import Image

from module.hologram import Hologram
from module.config import HoloConfig
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import os
import cv2
import time
import numpy as np
from numpy.fft import fftshift, fft2, ifft2, ifftshift
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from typing import Callable, Optional, Tuple, List, Dict, Any

from utils.utils_for_focusing.rcf.data_loader import prepare_image_PIL
from utils.utils_for_focusing.rcf.models.RCF import RCF
from utils.utils_for_focusing.rcf.rcf_predict import calculate_brightness_concentration


# ====================================================================
# PCHIP 贝叶斯优化寻峰器
# ====================================================================
class PeakFinder:
    """
    PCHIP 代理模型优化寻峰器

    通过少量采样点构建 PCHIP 插值代理模型，在代理模型上搜索最优解。
    支持 exploration（探索）和 exploitation（利用）两阶段自适应采样。

    Parameters
    ----------
    func : Callable
        待优化的目标函数，接受 x 和 scale 参数，返回标量值
    lb : float
        搜索区间下界
    ub : float
        搜索区间上界
    max_evals : int
        最大评估次数
    target : str
        优化目标：'min' 或 'max'
    precision_factor : float
        精度因子，越大越精度越高
    initial_points_factor : float
        初始采样点数因子
    scale : optional
        传递给 func 的额外参数
    seed : int
        随机种子，初始化时固定 np.random 状态，保证 CPU/GPU 采样序列一致、结果可复现
    """

    def __init__(self, func: Callable, lb: float = -0.00060, ub: float = 0.00060,
                 max_evals: int = 50, target: str = 'min',
                 precision_factor: float = 1.0, initial_points_factor: float = 1.0,
                 scale=None, seed: int = 42):
        # 固定随机种子，保证 CPU/GPU 两次运行的 z 采样序列一致，结果可复现、可对比
        np.random.seed(seed)
        self.func = func
        self.lb = lb
        self.ub = ub
        self.max_evals = max_evals
        self.target = target
        self.x_history: List[float] = []
        self.y_history: List[float] = []
        self.eval_count = 0
        self.phase = 'exploration'
        self.precision_factor = precision_factor
        self.initial_points_factor = initial_points_factor
        self.scale = scale
        self.seed = seed
        self.success = False  # 寻峰成功标志

    def evaluate(self, x: float) -> float:
        """评估目标函数"""
        if self.eval_count >= self.max_evals:
            return float('inf') if self.target == 'min' else float('-inf')

        # 检查重复
        for i, prev_x in enumerate(self.x_history):
            if abs(prev_x - x) < 1e-12:
                return self.y_history[i]

        # clip 到边界
        x = np.clip(x, self.lb, self.ub)
        y = self.func(x, scale=self.scale)

        self.x_history.append(x)
        self.y_history.append(y)
        self.eval_count += 1
        return y

    def get_current_best(self) -> Tuple[Optional[float], Optional[float]]:
        """获取当前最佳点"""
        if not self.x_history:
            return None, None
        if self.target == 'min':
            best_idx = np.argmin(self.y_history)
        else:
            best_idx = np.argmax(self.y_history)
        return self.x_history[best_idx], self.y_history[best_idx]

    def build_surrogate_model(self) -> Optional[PchipInterpolator]:
        """构建 PCHIP 代理模型"""
        if len(self.x_history) < 4:
            return None

        x_array = np.array(self.x_history)
        y_array = np.array(self.y_history)

        sorted_indices = np.argsort(x_array)
        x_sorted = x_array[sorted_indices]
        y_sorted = y_array[sorted_indices]

        try:
            return PchipInterpolator(x_sorted, y_sorted, extrapolate=False)
        except Exception:
            return None

    def find_surrogate_optimum(self, surrogate_model: PchipInterpolator) -> Optional[float]:
        """在代理模型上寻找最优点"""
        if surrogate_model is None:
            return None
        try:
            if self.target == 'min':
                objective = lambda x: surrogate_model(x)
            else:
                objective = lambda x: -surrogate_model(x)

            result = minimize_scalar(
                objective,
                bounds=(self.lb, self.ub),
                method='bounded',
                options={'xatol': 1e-11}
            )
            if result.success:
                return result.x
            else:
                return None
        except Exception:
            return None

    def acquisition_function(self, x_candidates: np.ndarray) -> np.ndarray:
        """采集函数 - 平衡探索与利用"""
        current_best_x, current_best_y = self.get_current_best()
        if current_best_x is None:
            return np.zeros(len(x_candidates))

        scores = []
        range_width = self.ub - self.lb
        for x in x_candidates:
            distance_to_best = abs(x - current_best_x) / range_width
            score = 1 - distance_to_best
            scores.append(score)
        return np.array(scores)

    def adaptive_sampling_strategy(self) -> float:
        """自适应采样策略"""
        current_best_x, _ = self.get_current_best()
        candidates = []

        if self.phase == 'exploration':
            n_points = max(15, int(20 * self.initial_points_factor))
            n_uniform = int(n_points * 0.7)
            n_centered = n_points - n_uniform

            uniform_samples = np.random.uniform(self.lb, self.ub, n_uniform)
            center = (self.lb + self.ub) / 2
            std = (self.ub - self.lb) / 6
            centered_samples = np.random.normal(center, std, n_centered)
            centered_samples = np.clip(centered_samples, self.lb, self.ub)
            candidates = np.concatenate([uniform_samples, centered_samples]).tolist()

        else:
            surrogate = self.build_surrogate_model()
            if surrogate and current_best_x is not None:
                opt = self.find_surrogate_optimum(surrogate)
                if opt is not None:
                    candidates.append(opt)

            radius = 0.15 * (self.ub - self.lb)
            n_points = max(10, int(12 * self.initial_points_factor))
            for _ in range(n_points):
                candidate = current_best_x + radius * np.random.uniform(-1, 1)
                candidates.append(candidate)

            if len(self.y_history) >= 5:
                y_array = np.array(self.y_history)
                if self.target == 'min':
                    top_indices = np.argsort(y_array)[:3]
                else:
                    top_indices = np.argsort(y_array)[-3:]
                for idx in top_indices[1:]:
                    x_secondary = self.x_history[idx]
                    if abs(x_secondary - current_best_x) > 0.1 * (self.ub - self.lb):
                        for _ in range(3):
                            candidate = x_secondary + 0.05 * (self.ub - self.lb) * np.random.uniform(-1, 1)
                            candidates.append(candidate)

        if not candidates:
            n_points = max(10, int(12 * self.initial_points_factor))
            candidates = np.random.uniform(self.lb, self.ub, n_points).tolist()

        candidates = [np.clip(c, self.lb, self.ub) for c in candidates]
        if len(candidates) > 0:
            scores = self.acquisition_function(np.array(candidates))
            best_idx = np.argmax(scores)
            return candidates[best_idx]
        else:
            return (self.lb + self.ub) / 2

    def update_phase(self):
        """更新搜索阶段"""
        n_evals = self.eval_count
        if n_evals < self.max_evals * 0.35:
            self.phase = 'exploration'
        elif n_evals < self.max_evals * 0.75:
            self.phase = 'exploitation'
        else:
            self.phase = 'refinement'

    def calculate_surrogate_fitting_error(self, surrogate_model: PchipInterpolator,
                                           error_metric: str = 'rmse') -> Tuple[float, float]:
        """
        计算代理模型与实际函数的拟合误差
        Returns: (error_value, error_ratio)
        """
        if surrogate_model is None or len(self.y_history) < 4:
            return float('inf'), 1.0

        x_array = np.array(self.x_history)
        y_array = np.array(self.y_history)

        y_pred = surrogate_model(x_array)
        errors = np.abs(y_array - y_pred)

        if error_metric == 'rmse':
            error_value = np.sqrt(np.mean(errors ** 2))
        elif error_metric == 'mae':
            error_value = np.mean(errors)
        elif error_metric == 'max':
            error_value = np.max(errors)
        elif error_metric == 'relative':
            y_range = np.max(y_array) - np.min(y_array)
            error_value = np.max(errors) / (y_range if y_range > 1e-12 else 1e-12)
        else:
            error_value = np.mean(errors)

        y_range = np.max(y_array) - np.min(y_array)
        error_ratio = error_value / (y_range if y_range > 1e-12 else 1e-12)
        return error_value, error_ratio

    def judge_success(self, surrogate_error_threshold: float = 0.15) -> Tuple[bool, str, Dict]:
        """
        综合判定寻峰是否成功
        """
        f_values = np.array(self.y_history)
        surrogate = self.build_surrogate_model()

        metrics = {
            'eval_count': self.eval_count,
            'data_points': len(f_values),
            'surrogate_error': None,
            'error_ratio': None,
            'y_range': np.max(f_values) - np.min(f_values)
        }

        if len(f_values) < 10:
            return False, "数据点不足（<10个）", metrics

        if surrogate is None:
            return False, "代理模型构建失败", metrics

        error_value, error_ratio = self.calculate_surrogate_fitting_error(
            surrogate, error_metric='max'
        )
        metrics['surrogate_error'] = error_value
        metrics['error_ratio'] = error_ratio

        if error_ratio > surrogate_error_threshold:
            return False, f"代理模型拟合误差过大 (误差比: {error_ratio:.4f} > {surrogate_error_threshold})", metrics

        if metrics['y_range'] < 1e-12:
            return False, "函数值范围过小，可能是噪声", metrics

        return True, "寻峰成功", metrics

    def find_optimum(self) -> float:
        """主搜索算法，返回最优点 x"""
        initial_points_num = max(5, int(7 * self.initial_points_factor))

        # 初始分段采样
        for i in range(initial_points_num):
            segment_start = self.lb + i * (self.ub - self.lb) / initial_points_num
            segment_end = self.lb + (i + 1) * (self.ub - self.lb) / initial_points_num
            point = np.random.uniform(segment_start, segment_end)
            self.evaluate(point)

        # 添加中心点
        self.evaluate((self.lb + self.ub) / 2)

        stagnation_count = 0
        prev_best_y = float('inf') if self.target == 'min' else float('-inf')

        while self.eval_count < self.max_evals:
            self.update_phase()
            current_best_x, current_best_y = self.get_current_best()

            next_x = self.adaptive_sampling_strategy()
            self.evaluate(next_x)

            if current_best_y is not None:
                improvement = abs(current_best_y - prev_best_y)
                convergence_threshold = 1e-4 / self.precision_factor
                if improvement < convergence_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                prev_best_y = current_best_y

                if stagnation_count >= 8:
                    break

        final_x, _ = self.get_current_best()
        self.success, reason, metrics = self.judge_success(surrogate_error_threshold=0.15)

        return final_x


# ====================================================================
# PCHIP 快速自聚焦主类
# ====================================================================
class FastFocusPCHIP:
    """
    基于 PCHIP 的快速自聚焦类

    功能：
    - 在 z 搜索范围内通过少量采样重建，用 PCHIP 插值构建清晰度代理模型
    - 在代理模型上搜索最优聚焦深度 z
    - 仅需一次最终重建即可得到全聚焦图
    - 假设所有颗粒在同一聚焦面

    聚焦分数计算方式：每个 z 做一次角谱法重建 → 重建振幅图送入 RCF 模型
    → RCF 输出边缘响应图 → 计算边缘响应图的亮度集中度（非零像素方差）
    → 该值作为该 z 位置的聚焦分数。

    Parameters
    ----------
    hologram : Hologram
        Hologram 实例，需包含 .spectrum 属性
    config : HoloConfig
        配置实例
    rcf_model : RCF, optional
        外部传入的 RCF 模型实例，避免重复加载
    device : torch.device, optional
        推理设备
    k_size : int
        RCF 输入图像缩放因子（默认 8 与原有方法一致）
    """

    _rcf_model_loaded = False
    _rcf_model = None
    _use_gpu = False  # CPU 版全程 CPU；GPU 子类覆盖为 True

    def __init__(self, hologram: Hologram, config: HoloConfig,
                 rcf_model=None, device=None, k_size: int = 8):

        self._hologram = hologram
        self._config = config

        # 从 config 获取参数
        self.spectrum = hologram.spectrum
        self.pixel_size = config.image_info['pixel_size'] * unit_um
        self.wavelength = config.image_info['wavelength'] * unit_nm

        # 搜索范围与 PCHIP 参数
        self.z_start = config.fast_focus['z_start'] * unit_mm
        self.z_end = config.fast_focus['z_end'] * unit_mm
        self.max_evals = config.fast_focus['max_evals']
        self.initial_points_factor = config.fast_focus['initial_points_factor']
        self.k_size = k_size

        # 设备：CPU 版全程 CPU，GPU 版强制 cuda，均不读 config.focusing['device']，
        # 避免为切 CPU 版而改 focusing.device 时把 GPU 版也拖到 CPU。
        if getattr(self, '_use_gpu', False):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # RCF 模型加载
        if rcf_model is not None:
            self.rcf_model = rcf_model
        elif not FastFocusPCHIP._rcf_model_loaded:
            self.rcf_model = self._load_rcf_model()
            FastFocusPCHIP._rcf_model = self.rcf_model
            FastFocusPCHIP._rcf_model_loaded = True
        else:
            self.rcf_model = FastFocusPCHIP._rcf_model

        self.rcf_model.to(self.device)
        self.rcf_model.eval()

        # 预计算结果
        self._optimal_z: Optional[float] = None
        self._optimal_image: Optional[np.ndarray] = None
        self._z_history: List[float] = []
        self._metric_history: List[float] = []

        # 频谱相关预计算
        self._precompute_frequency()

    def _load_rcf_model(self):
        """加载 RCF 模型"""
        rcf_model = RCF(self.device)
        rcf_model.to(self.device)
        rcf_path = self._config.focusing['rcf_model_path']
        checkpoint = torch.load(rcf_path, map_location=self.device)
        rcf_model.load_state_dict(checkpoint['state_dict'])
        return rcf_model

    def _precompute_frequency(self):
        """预计算频域网格，并对频谱做低频裁剪以加速重建"""
        scale = self.k_size
        M_full, N_full = self.spectrum.shape

        # 裁剪频谱中心低频区域
        M_crop = M_full // scale
        N_crop = N_full // scale
        M_start = (M_full - M_crop) // 2
        N_start = (N_full - N_crop) // 2
        self.spectrum_cropped = self.spectrum[M_start:M_start + M_crop, N_start:N_start + N_crop]
        self.pixel_size_scaled = self.pixel_size * scale

        # 波前（从裁剪后的频谱逆变换得到）
        self.wavefront = ifft2(ifftshift(self.spectrum_cropped))
        M, N = self.wavefront.shape

        # 空间频率坐标（使用缩放后的像素尺寸）
        self.fft_x = fftshift(np.fft.fftfreq(N, d=self.pixel_size_scaled))
        self.fft_y = fftshift(np.fft.fftfreq(M, d=self.pixel_size_scaled))
        self.fft_mesh_x, self.fft_mesh_y = np.meshgrid(self.fft_x, self.fft_y)
        self.fft_squa = self.fft_mesh_x ** 2 + self.fft_mesh_y ** 2

        # 保存全尺寸信息用于最终输出
        self._M_full = M_full
        self._N_full = N_full

    def _angular_spectrum_propagate(self, z: float) -> np.ndarray:
        """
        角谱法单截面重建（在缩放后的尺寸上进行）

        Parameters
        ----------
        z : float
            传播距离（单位：米）

        Returns
        -------
        np.ndarray
            重建后的振幅图像（缩小后的尺寸）
        """
        # 传播函数（角谱传递函数）
        H = np.exp(1j * 2 * np.pi / self.wavelength * z) * \
            np.exp(-1j * np.pi * self.wavelength * z * self.fft_squa)

        # 频域传播（使用裁剪后的频谱）
        A = H * self.spectrum_cropped
        U = ifft2(ifftshift(A))

        # 返回振幅
        return np.abs(U)

    def _rcf_predict_single(self, amplitude: np.ndarray) -> float:
        """
        对单张振幅图做 RCF 边缘预测，返回聚焦分数

        Parameters
        ----------
        amplitude : np.ndarray
            重建振幅图（已经是 1/k_size 尺寸）

        Returns
        -------
        float
            聚焦分数 = 边缘响应图的亮度集中度（非零像素方差）
            越大表示越聚焦
        """
        # 1. 准备输入：灰度图 → RGB 三通道，转为 float32
        #    注意：重建已经是在缩小后的尺寸上做的，无需再次 resize
        img_rgb = cv2.cvtColor(amplitude, cv2.COLOR_GRAY2RGB).astype(np.float32)
        img_np = prepare_image_PIL(img_rgb)  # (3, H, W), float32 numpy array
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 2. RCF 推理
        with torch.no_grad():
            results = self.rcf_model(img_tensor)
            result = torch.squeeze(results[-1].detach()).cpu().numpy()

        # 3. 计算亮度集中度
        result_pil = Image.fromarray((result * 255).astype(np.uint8))

        # 保存 RCF edge map 到 tmp 目录
        os.makedirs('tmp/rcf_results', exist_ok=True)
        timestamp = time.time()
        result_pil.save(f'tmp/rcf_results/rcf_edge_{timestamp:.4f}.png')

        concentration = calculate_brightness_concentration(np.array(result_pil))

        return concentration

    def _evaluate_at_z(self, z: float, scale=None) -> float:
        """
        在给定 z 处重建 + RCF 边缘预测 + 计算聚焦分数

        Parameters
        ----------
        z : float
            传播距离（米）
        scale : optional
            保留参数，兼容 PeakFinder 接口

        Returns
        -------
        float
            负聚焦分数（PeakFinder 默认 target='min'，所以返回负值）
        """
        # 1. 单截面重建
        image = self._angular_spectrum_propagate(z)

        # 2. 归一化到 [0,1] 供 RCF 使用
        abs_v = np.abs(image)
        if abs_v.max() > abs_v.min():
            normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
        else:
            normalized = abs_v
        img_uint8 = (normalized * 255).astype(np.uint8)

        # 3. RCF 边缘预测 + 计算聚焦分数
        concentration = self._rcf_predict_single(img_uint8)

        # 4. 记录历史
        self._z_history.append(z)
        self._metric_history.append(concentration)

        return -concentration  # 取负值，因为 PeakFinder 默认最小化

    def run(self) -> Tuple[np.ndarray, float]:
        """
        执行 PCHIP 快速自聚焦

        Returns
        -------
        Tuple[np.ndarray, float]
            (最优聚焦图像, 最优 z 值)
        """
        # 创建寻峰器
        finder = PeakFinder(
            func=self._evaluate_at_z,
            lb=self.z_start,
            ub=self.z_end,
            max_evals=self.max_evals,
            target='min',  # 因为我们返回的是负聚焦分数
            initial_points_factor=self.initial_points_factor,
        )

        # 搜索最优 z
        start_time = time.time()
        optimal_z = finder.find_optimum()
        search_time = time.time() - start_time

        # 在最优 z 处做最终重建（小尺寸）
        self._optimal_z = optimal_z
        self._optimal_image = self._angular_spectrum_propagate(optimal_z)

        # 归一化到 0-255
        abs_v = np.abs(self._optimal_image)
        if abs_v.max() > abs_v.min():
            normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
        else:
            normalized = abs_v
        self._optimal_image = (normalized * 255).astype(np.uint8)

        # 缩放到全尺寸用于后续处理和显示
        full_size = (self._N_full, self._M_full)
        self._optimal_image_full = cv2.resize(self._optimal_image, full_size,
                                               interpolation=cv2.INTER_LINEAR)

        # 状态信息
        self._hologram.status_msg = (
            f"Fast Focus (PCHIP) Done. "
            f"Optimal z = {optimal_z / unit_mm:.4f} mm, "
            f"Evals = {finder.eval_count}, "
            f"Time = {search_time:.2f}s"
        )

        print('opt z', optimal_z)
        return self._optimal_image_full, optimal_z

    def modify_hologram_and_config(self):
        """更新 hologram 对象（使用全尺寸图像）"""
        self._hologram.focusing = self._optimal_image_full
        self._hologram.focusing_z = [self._optimal_z]
        self._hologram.focusing_each = {}

    def get_history(self) -> Tuple[List[float], List[float]]:
        """获取搜索历史"""
        return self._z_history, self._metric_history


# ====================================================================
# GPU 加速版本（可选）
# ====================================================================
class FastFocusPCHIP_GPU(FastFocusPCHIP):
    """
    基于 PCHIP 的快速自聚焦类（GPU 加速版）

    使用 PyTorch 加速角谱法重建过程，RCF 推理同样使用 PyTorch。
    不再使用 CuPy，避免与 PyTorch 混用导致堆内存损坏 (0xC0000374)。
    CuPy 版本见下方注释备份。
    """
    _use_gpu = True

    def _precompute_frequency(self):
        """预计算频域网格（GPU），做低频裁剪加速。torch 版本。"""
        scale = self.k_size
        M_full, N_full = self.spectrum.shape
        M_crop = M_full // scale
        N_crop = N_full // scale
        M_start = (M_full - M_crop) // 2
        N_start = (N_full - N_crop) // 2

        # 频谱裁剪 + 移到 GPU
        spectrum_cropped = self.spectrum[M_start:M_start + M_crop, N_start:N_start + N_crop]
        self.spectrum_gpu = torch.as_tensor(spectrum_cropped, device=self.device, dtype=torch.complex64)
        self.wavefront_gpu = torch.fft.ifft2(torch.fft.ifftshift(self.spectrum_gpu))

        self.pixel_size_scaled = self.pixel_size * scale

        M, N = self.wavefront_gpu.shape
        fft_x = torch.fft.fftshift(torch.fft.fftfreq(N, d=self.pixel_size_scaled, device=self.device))
        fft_y = torch.fft.fftshift(torch.fft.fftfreq(M, d=self.pixel_size_scaled, device=self.device))
        # cupy.meshgrid 默认 'xy' 索引，torch 需显式指定以保持一致
        fft_mesh_x, fft_mesh_y = torch.meshgrid(fft_x, fft_y, indexing='xy')
        self.fft_squa_gpu = fft_mesh_x ** 2 + fft_mesh_y ** 2

        # 保存全尺寸信息用于最终输出
        self._M_full = M_full
        self._N_full = N_full

    def _angular_spectrum_propagate(self, z: float) -> np.ndarray:
        """GPU 加速的单截面重建（torch 版本）"""
        wavelength = self.wavelength
        # 传播函数（角谱传递函数）：exp(1j*(2π/λ*z - π*λ*z*f²))
        H = torch.exp(1j * (2 * torch.pi / wavelength * z
                            - torch.pi * wavelength * z * self.fft_squa_gpu))

        A = H * self.spectrum_gpu
        U = torch.fft.ifft2(torch.fft.ifftshift(A))

        return torch.abs(U).detach().cpu().numpy()

    # '''---------------- CuPy 备份版本（保留以备需要时切换）----------------
    # 注意：cupy 与 torch 不可混用，会导致堆内存损坏 (0xC0000374)。
    # def _precompute_frequency(self):
    #     """预计算频域网格（GPU），做低频裁剪加速"""
    #     import cupy as cp
    #
    #     self.cp = cp
    #     scale = self.k_size
    #     M_full, N_full = self.spectrum.shape
    #     M_crop = M_full // scale
    #     N_crop = N_full // scale
    #     M_start = (M_full - M_crop) // 2
    #     N_start = (N_full - N_crop) // 2
    #
    #     # 频谱裁剪 + 移到 GPU
    #     spectrum_cropped = self.spectrum[M_start:M_start + M_crop, N_start:N_start + N_crop]
    #     self.spectrum_gpu = cp.asarray(spectrum_cropped)
    #     self.wavefront_gpu = cp.fft.ifft2(cp.fft.ifftshift(self.spectrum_gpu))
    #
    #     self.pixel_size_scaled = self.pixel_size * scale
    #
    #     M, N = self.wavefront_gpu.shape
    #     fft_x = cp.fft.fftshift(cp.fft.fftfreq(N, d=self.pixel_size_scaled))
    #     fft_y = cp.fft.fftshift(cp.fft.fftfreq(M, d=self.pixel_size_scaled))
    #     fft_mesh_x, fft_mesh_y = cp.meshgrid(fft_x, fft_y)
    #     self.fft_squa_gpu = fft_mesh_x ** 2 + fft_mesh_y ** 2
    #
    #     self.wavelength_gpu = cp.asarray(self.wavelength)
    #
    #     # 保存全尺寸信息用于最终输出
    #     self._M_full = M_full
    #     self._N_full = N_full
    #
    # def _angular_spectrum_propagate(self, z: float) -> np.ndarray:
    #     """GPU 加速的单截面重建"""
    #     cp = self.cp
    #
    #     H = cp.exp(1j * 2 * cp.pi / self.wavelength_gpu * z) * \
    #         cp.exp(-1j * cp.pi * self.wavelength_gpu * z * self.fft_squa_gpu)
    #
    #     A = H * self.spectrum_gpu
    #     U = cp.fft.ifft2(cp.fft.ifftshift(A))
    #
    #     return cp.asnumpy(cp.abs(U))
    # -------------------------------------------------------------------'''

    def _rcf_predict_single(self, amplitude: np.ndarray) -> float:
        """RCF 边缘预测（已在 GPU 上，与 CPU 版相同）"""
        return super()._rcf_predict_single(amplitude)


if __name__ == '__main__':
    print('Utils FastFocusPCHIP Module', end='\n\n')

    # 测试用例
    import matplotlib.pyplot as plt

    # 测试 1：单峰函数
    print("=" * 70)
    print("测试1：单峰函数 - (x - 0.0002)^2")
    print("=" * 70)

    def single_peak(x, scale=None):
        return (x - 0.0002) ** 2

    finder = PeakFinder(
        func=single_peak,
        lb=-0.0006,
        ub=0.0006,
        max_evals=50,
        target='min'
    )
    optimal_x = finder.find_optimum()
    print(f"最优 x = {optimal_x:.8f} (理论值: 0.00020000)")
    print(f"成功: {finder.success}")
    print(f"评估次数: {finder.eval_count}")

    # 测试 2：聚焦评价指标测试
    print("\n" + "=" * 70)
    print("测试2：FocusMetric 函数测试")
    print("=" * 70)

    test_img = np.random.rand(100, 100).astype(np.float32)
    # print(f"方差: {FocusMetric.variance(test_img):.4f}")
    # print(f"梯度幅值: {FocusMetric.gradient_magnitude(test_img):.4f}")
    # print(f"拉普拉斯方差: {FocusMetric.laplacian_variance(test_img):.4f}")

    print("\nFastFocusPCHIP module loaded successfully.")
