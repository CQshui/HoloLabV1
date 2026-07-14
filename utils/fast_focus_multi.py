"""
Multi-Particle PCHIP Fast Autofocusing Module

多颗粒快速自聚焦模块：
- 适用于颗粒分布在多个不同 z 深度的情况
- 先用少量全局 z 平面初始化 YOLO+SORT 轨迹
- 为每个颗粒建立独立的 PCHIP 代理模型
- 轮询：所有颗粒请求的新 z 平面合并去重后统一批量重建
- 利用卡尔曼滤波预测位置，局部裁剪计算清晰度，跳过后续 YOLO
"""
import os
import cv2
import time
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict, Set
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader

from module.hologram import Hologram
from module.config import HoloConfig
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

from utils.utils_for_focusing.sort.kalman_trace_iou_center import Kalman_tracker, Sort
from utils.utils_for_focusing.yoloV8.yolo import YOLO
from utils.utils_for_focusing.rcf.models.RCF import RCF
from utils.utils_for_focusing.rcf.data_loader import prepare_image_PIL, convert_to_rgb
from utils.utils_for_focusing.rcf.rcf_predict import calculate_brightness_concentration
from utils.utils_for_focusing.final import RCFLoader, rcf_predict_V1
from utils.reconstruction import Reconstruction


# ====================================================================
# 轻量级 PCHIP 寻峰器（每个颗粒独立使用）
# ====================================================================
class ParticlePeakFinder:
    """
    单个颗粒的 PCHIP 寻峰器

    记录该颗粒在所有已评估 z 平面上的聚焦分数，
    用 PCHIP 构建代理模型，预测下一个采样点。
    """

    def __init__(self, particle_id: int, z_range: Tuple[float, float],
                 max_evals: int = 80, convergence_threshold: float = 1e-6):
        self.particle_id = particle_id
        self.z_min, self.z_max = z_range
        self.max_evals = max_evals
        self.convergence_threshold = convergence_threshold  # 归一化收敛阈值

        self.z_history: List[float] = []
        self.score_history: List[float] = []
        self.eval_count = 0
        self.best_z: Optional[float] = None
        self.best_score: Optional[float] = None
        self.converged = False

    def add_evaluation(self, z: float, score: float):
        """添加一次评估结果"""
        self.z_history.append(z)
        self.score_history.append(score)
        self.eval_count += 1

        # 更新最优
        if self.best_z is None or score > self.best_score:
            self.best_z = z
            self.best_score = score

    def build_surrogate(self) -> Optional[PchipInterpolator]:
        """构建 PCHIP 代理模型"""
        if len(self.z_history) < 4:
            return None
        x = np.array(self.z_history)
        y = np.array(self.score_history)
        idx = np.argsort(x)
        try:
            return PchipInterpolator(x[idx], y[idx], extrapolate=False)
        except Exception:
            return None

    def suggest_next_z(self) -> Optional[float]:
        """基于当前代理模型，建议下一个最有信息量的 z"""
        surrogate = self.build_surrogate()
        if surrogate is None:
            # 初始阶段：在搜索范围内均匀采样
            if len(self.z_history) == 0:
                return (self.z_min + self.z_max) / 2
            else:
                # 在已采样点之间的最大间隙处采样
                sorted_z = sorted(self.z_history)
                gaps = [(sorted_z[i+1] - sorted_z[i], (sorted_z[i+1] + sorted_z[i]) / 2)
                        for i in range(len(sorted_z)-1)]
                if len(self.z_history) >= 3 and self.z_min < sorted_z[0]:
                    gaps.append((sorted_z[0] - self.z_min, (self.z_min + sorted_z[0]) / 2))
                if self.z_max > sorted_z[-1]:
                    gaps.append((self.z_max - sorted_z[-1], (self.z_max + sorted_z[-1]) / 2))
                if gaps:
                    return max(gaps, key=lambda x: x[0])[1]
                return (self.z_min + self.z_max) / 2

        # exploitation: 在代理模型上找最优
        try:
            result = minimize_scalar(
                lambda x: -surrogate(x),
                bounds=(self.z_min, self.z_max),
                method='bounded'
            )
            if result.success:
                predicted_best = result.x
                # 检查是否收敛
                if self.best_z is not None:
                    delta = abs(predicted_best - self.best_z) / (self.z_max - self.z_min)
                    if delta < self.convergence_threshold:
                        self.converged = True
                        return None
                return predicted_best
        except Exception:
            pass

        # fallback: 在最佳点附近随机扰动
        if self.best_z is not None:
            return self.best_z + 0.02 * (self.z_max - self.z_min) * np.random.uniform(-1, 1)
        return (self.z_min + self.z_max) / 2


# ====================================================================
# 多颗粒 PCHIP 自聚焦主类
# ====================================================================
class MultiFocusPCHIP:
    """
    多颗粒 PCHIP 自聚焦

    流程：
    1. 初始阶段：在少量全局 z 平面做全尺寸重建 → YOLO+SORT 建立轨迹
    2. 为每个颗粒创建 PCHIP 代理模型
    3. 轮询：所有颗粒请求新 z → 合并去重 → 批量重建
    4. 卡尔曼预测当前位置，局部裁剪计算清晰度，跳过后续 YOLO
    5. 重复直到所有颗粒收敛或达到最大轮次
    """

    def __init__(self, hologram, config):
        self._hologram = hologram
        self._config = config

        # 基本参数
        self.pixel_size = config.image_info['pixel_size'] * unit_um
        self.wavelength = config.image_info['wavelength'] * unit_nm
        self.z_start = config.reconstruction['z_start'] * unit_mm
        self.z_end = config.reconstruction['z_end'] * unit_mm
        self.z_step = config.reconstruction['z_step'] * unit_mm
        self.z_num = int((self.z_end - self.z_start) / self.z_step) + 1
        self.z_array = np.linspace(self.z_start, self.z_end, self.z_num)
        self.z_list = self.z_array.tolist()

        self.device = torch.device(config.focusing.get('device', 'cuda'))
        self.gpu_id = config.focusing.get('gpu_id', 0)
        self.cpu_num = config.focusing.get('cpu_num', 1)
        self.rcf_scale = config.focusing.get('rcf_scale', 8)

        # YOLO + RCF 模型
        self.yoloModel = None
        self.rcfModel = None
        self._load_models()

        # 结果
        self.focusing = None            # 全尺寸聚焦图像
        self.focusing_z = []            # 每个颗粒的聚焦 z
        self.focusing_xy = []           # 每个颗粒的 (x1, y1, x2, y2)
        self.focusing_each = {}         # 每个颗粒的聚焦子图

        # 多颗粒 PCHIP 配置
        self.initial_global_planes = config.fast_focus_multi.get('initial_global_planes', 5)
        self.max_pchip_rounds = config.fast_focus_multi.get('max_pchip_rounds', 50)
        self.max_evals_per_particle = config.fast_focus_multi.get('max_evals_per_particle', 12)
        self.crop_margin = config.fast_focus_multi.get('crop_margin', 20)  # 局部裁剪边距

    def _load_models(self):
        """加载 YOLO 和 RCF 模型"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        # YOLO
        yolo_path = self._config.focusing.get('yolo_model_path', '')
        CUDA_Available = (self.device.type == 'cuda')

        if os.path.exists(yolo_path):
            self.yoloModel = YOLO(input_shape=[640, 640], phi='s',
                                  model_path=yolo_path,
                                  cuda=CUDA_Available,
                                  letterbox_image=True,
                                  confidence=0.5,
                                  nms_iou=0.3,
                                  device=self.device
                                  )
            # self.yoloModel.load_weights(yolo_path)
            # self.yoloModel.to(self.device)

        # RCF
        from utils.utils_for_focusing.rcf.models.RCF import RCF
        rcf_path = self._config.focusing.get('rcf_model_path', '')
        if os.path.exists(rcf_path):
            self.rcfModel = RCF(self.device)
            checkpoint = torch.load(rcf_path, map_location=self.device)
            self.rcfModel.load_state_dict(checkpoint['state_dict'])
            self.rcfModel.to(self.device)
            self.rcfModel.eval()

    def _batch_reconstruct(self, z_values: List[float]) -> Dict[float, np.ndarray]:
        """
        对一组 z 值做批量重建
        返回 {z: 振幅图像(全尺寸)}
        """
        # 使用 Reconstruction 类做单截面重建
        from utils.reconstruction import Reconstruction
        recon = Reconstruction(self._hologram, self._config)
        recon.z_array = np.array(sorted(set(z_values)))
        recon.z_list = recon.z_array.tolist()
        recon.z_num = len(recon.z_array)
        recon.Reconstruction_Angular_Spectrum_CPU()

        result = {}
        for z, img in zip(recon.z_list, recon.reconstruction_list):
            result[z] = np.abs(img)
        return result

    def _crop_particle(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                       margin: int = 20) -> np.ndarray:
        """
        在图像中裁剪颗粒区域（带边距）

        Parameters
        ----------
        image : np.ndarray
            全尺寸重建振幅图
        bbox : (x1, y1, x2, y2)
            颗粒边界框（原始坐标）
        margin : int
            裁剪边距

        Returns
        -------
        np.ndarray
            裁剪后的颗粒子图
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        return image[y1:y2, x1:x2]

    def _rcf_score_on_crop(self, crop: np.ndarray) -> float:
        """
        对颗粒裁剪图做 RCF 推理，返回聚焦分数

        Parameters
        ----------
        crop : np.ndarray
            颗粒子图（灰度）

        Returns
        -------
        float
            聚焦分数（亮度集中度）
        """
        # 缩放到 RCF 输入尺寸
        h, w = crop.shape
        h_small, w_small = max(1, h), max(1, w)
        img_small = cv2.resize(crop, (w_small, h_small), interpolation=cv2.INTER_LINEAR)

        # 归一化到 0-255 uint8
        abs_v = np.abs(img_small)
        if abs_v.max() > abs_v.min():
            normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
        else:
            normalized = abs_v
        img_uint8 = (normalized * 255).astype(np.uint8)

        # RCF 推理
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32)
        img_np = prepare_image_PIL(img_rgb)  # (3, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            results = self.rcfModel(img_tensor)
            result = torch.squeeze(results[-1].detach()).cpu().numpy()

        # 聚焦分数
        # 【注意】避免在 GPU 计算流程中执行磁盘 I/O，可能引发时序问题
        result_pil = Image.fromarray((result * 255).astype(np.uint8))
        # result_pil.save(r'E:\Projects\HoloLabV1\tmp\rcf_results\{:.4f}.png'.format(time.time()))
        concentration = calculate_brightness_concentration(np.array(result_pil))
        return concentration

    def _run_yolo_sort_on_planes(self, plane_images: Dict[float, np.ndarray]) -> Dict[str, Dict]:
        """
        在多个 z 平面上运行 YOLO+SORT，建立颗粒轨迹

        Parameters
        ----------
        plane_images : {z: amplitude_image}
            初始全局 z 平面的重建图像

        Returns
        -------
        dict
            {particle_id: {'stacks': [...], 'positions': [...], 'bboxes': [...]}}
        """
        # 将每个平面的图像转为 uint8 用于 YOLO 检测
        z_list = sorted(plane_images.keys())
        image_stack_list = []
        for z in z_list:
            img = plane_images[z]
            abs_v = np.abs(img)
            if abs_v.max() > abs_v.min():
                normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
            else:
                normalized = abs_v
            image_stack_list.append((normalized * 255).astype(np.uint8))

        # 拼成 stack: (H, W, num_planes)
        image_stack = np.stack(image_stack_list, axis=2)

        # 用卡尔曼追踪
        mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
        tracker = Kalman_tracker(
            image_folder=None,
            output_folder=None,
            yolo_model=self.yoloModel,
            sort=mot_tracker,
            image_stack=image_stack,
            device=self.device
        )

        stacks, stacks_ori, names, positions = tracker.get_stacks(get_position=True)

        # 整理结果
        particles = {}
        for i, (stack, stack_ori, pos) in enumerate(zip(stacks, stacks_ori, positions)):
            particle_id = f"particle_{i}"
            # stack 是 (num_planes, H_crop, W_crop) 格式
            # 但我们需要每个平面的裁剪图对应到原始全尺寸坐标
            particles[particle_id] = {
                'stacks': [stack[j] for j in range(len(stack))],
                'positions': pos,  # (x1, y1, x2, y2) 在缩放尺寸上的坐标（对应重建图尺寸）
            }

        return particles, z_list

    def run(self):
        """
        执行多颗粒 PCHIP 自聚焦

        Returns
        -------
        Tuple[np.ndarray, List[float], List[Tuple]]
            (全尺寸聚焦图像, 各颗粒聚焦 z 列表, 各颗粒坐标列表)
        """
        start_time = time.time()

        # ======================
        # 阶段 1：初始全局平面重建 + YOLO+SORT 建立轨迹
        # ======================
        print(f"[MultiFocusPCHIP] Stage 1: Initial reconstruction on {self.initial_global_planes} global planes")

        # 均匀选取初始平面
        if self.initial_global_planes >= self.z_num:
            init_z = self.z_array
        else:
            indices = np.linspace(0, self.z_num - 1, self.initial_global_planes, dtype=int)
            init_z = self.z_array[indices]

        # 批量重建初始平面
        init_images = self._batch_reconstruct(init_z.tolist())

        # YOLO+SORT 建立轨迹
        particles, init_z_list = self._run_yolo_sort_on_planes(init_images)
        print(f"  Detected {len(particles)} particles")

        if len(particles) == 0:
            print("[MultiFocusPCHIP] No particles detected, aborting.")
            h, w = self._hologram.hologram.shape[:2]
            self.focusing = np.zeros((h, w), dtype=np.uint8)
            self.focusing_z = []
            self.focusing_xy = []
            self.focusing_each = {}
            return self.focusing, self.focusing_z, self.focusing_xy

        # ======================
        # 阶段 2：为每个颗粒建立 PCHIP 代理模型
        # ======================
        print(f"[MultiFocusPCHIP] Stage 2: PCHIP surrogate modeling")

        particle_finders = {}
        for pid, pdata in particles.items():
            finder = ParticlePeakFinder(
                particle_id=pid,
                z_range=(self.z_start, self.z_end),
                max_evals=self.max_evals_per_particle,
            )
            particle_finders[pid] = finder

        # 记录每个颗粒的卡尔曼预测器（用于后续位置预测）
        # 从 YOLO+SORT 的初始结果中获取每个颗粒的轨迹
        # positions 是 (x1, y1, x2, y2) 在初始各平面的位置
        particle_kalman_states = {}  # 简化：直接用最近位置作为预测

        # 对初始平面：从重建图像中裁剪颗粒子图 → RCF 评分
        print("[Init] Evaluating particles on initial planes...")
        for z in init_z_list:
            full_img = init_images[z]
            for pid, pdata in particles.items():
                pos = pdata['positions']
                crop = self._crop_particle(full_img, pos, margin=self.crop_margin)
                score = self._rcf_score_on_crop(crop)
                particle_finders[pid].add_evaluation(z, score)

                # ---- 打印初始评估 ----
                finder = particle_finders[pid]
                print(f"[Init] {pid}: z={z:.6f}, score={score:.4f} | "
                      f"best z={finder.best_z:.6f}, best score={finder.best_score:.4f}")

        # ======================
        # 阶段 3：迭代 PCHIP 搜索
        # ======================
        print(f"[MultiFocusPCHIP] Stage 3: Iterative PCHIP search")

        for round_idx in range(self.max_pchip_rounds):
            # 收集所有颗粒建议的新 z
            requested_z: Set[float] = set()
            particle_request_map: Dict[float, List[str]] = {}  # z → [particle_ids]

            for pid, finder in particle_finders.items():
                if finder.converged or finder.eval_count >= finder.max_evals:
                    continue
                next_z = finder.suggest_next_z()
                if next_z is None:
                    continue
                # 四舍五入到最近的 z_step 整数倍
                next_z_rounded = round(next_z / self.z_step) * self.z_step
                next_z_rounded = np.clip(next_z_rounded, self.z_start, self.z_end)
                requested_z.add(next_z_rounded)
                particle_request_map.setdefault(next_z_rounded, []).append(pid)

            if not requested_z:
                print(f"  Round {round_idx + 1}: All particles converged.")
                break

            print(f"  Round {round_idx + 1}: {len(requested_z)} unique z-planes requested")

            # 批量重建请求的 z 平面
            new_images = self._batch_reconstruct(list(requested_z))

            # 对每个请求的平面：用卡尔曼预测位置 → 局部裁剪 → RCF 评分
            for z, full_img in new_images.items():
                for pid in particle_request_map.get(z, []):
                    pdata = particles[pid]

                    # 用卡尔曼预测当前位置
                    # 简化：用最近已知位置 + 线性插值
                    pos = pdata['positions']
                    predicted_pos = pos  # TODO: 用真正的卡尔曼预测

                    # 局部裁剪计算清晰度
                    crop = self._crop_particle(full_img, predicted_pos, margin=self.crop_margin)
                    score = self._rcf_score_on_crop(crop)

                    old_best_z = particle_finders[pid].best_z
                    old_best_score = particle_finders[pid].best_score
                    particle_finders[pid].add_evaluation(z, score)

                    # ---- 打印本次迭代评估 ----
                    finder = particle_finders[pid]
                    best_changed = (old_best_z != finder.best_z)
                    change_str = " -> best updated!" if best_changed else ""
                    print(f"  [Round {round_idx+1}] {pid}: z={z:.6f}, score={score:.4f} | "
                          f"best z={finder.best_z:.6f}, best score={finder.best_score:.4f}{change_str}")

        # ======================
        # 阶段 4：输出最终结果（解决 rcf_scale 缩放不一致问题）
        # ======================
        print(f"[MultiFocusPCHIP] Stage 4: Final output")

        # ---- 打印每个颗粒的完整评估历史 ----
        print("\n=== Particle Score History ===")
        for pid, finder in particle_finders.items():
            z_hist = np.array(finder.z_history)
            s_hist = np.array(finder.score_history)
            # 按 z 排序显示
            idx = np.argsort(z_hist)
            sorted_z = z_hist[idx]
            sorted_s = s_hist[idx]
            print(f"{pid}: best z={finder.best_z:.6f}, best score={finder.best_score:.4f}")
            print(f"  evaluations: z={sorted_z.tolist()}, scores={[f'{s:.4f}' for s in sorted_s]}")
            if finder.converged:
                print(f"  status: converged")
            else:
                print(f"  status: max evals reached" if finder.eval_count >= finder.max_evals else "  status: stopped")
        print("============================\n")

        # 注意：_batch_reconstruct 返回的图像是经过频谱裁剪后的缩放尺寸（1/rcf_scale），
        # YOLO+SORT 检测到的位置 coordinates 也对应这个缩放尺寸（因为 YOLO 运行在缩放图上）。
        # 因此最终输出时采用方案：在缩放尺寸下拼接，最后统一 resize 回全尺寸。

        scale = self.rcf_scale
        h_full, w_full = self._hologram.hologram.shape[:2]
        h_small, w_small = h_full // scale, w_full // scale

        # 先在缩放尺寸下拼接
        focusing_small = np.zeros((h_small, w_small), dtype=np.uint8)
        self.focusing_z = []
        self.focusing_xy = []
        self.focusing_each = {}

        # 收集所有颗粒的最佳 z 并去重排序
        best_zs = set()
        particle_final_info = []
        for pid, finder in particle_finders.items():
            if finder.best_z is not None:
                best_z = finder.best_z
                best_zs.add(best_z)
                particle_final_info.append((pid, best_z, particles[pid]))

        # 统一重建所有最优 z 平面
        if best_zs:
            final_images = self._batch_reconstruct(list(best_zs))

            for pid, best_z, pdata in particle_final_info:
                full_img = final_images.get(best_z)
                if full_img is None:
                    continue

                pos = pdata['positions']  # 缩放尺寸下的坐标 (x1, y1, x2, y2)

                # 将裁剪图放在缩放尺寸对应位置
                x1, y1, x2, y2 = pos
                crop = self._crop_particle(full_img, pos, margin=0)

                # 归一化到 uint8
                abs_v = np.abs(crop)
                if abs_v.max() > abs_v.min():
                    normalized = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
                else:
                    normalized = abs_v
                crop_uint8 = (normalized * 255).astype(np.uint8)

                focusing_small[y1:y2, x1:x2] = crop_uint8
                self.focusing_z.append(best_z)
                # 保存全尺寸坐标（供外部使用）
                full_pos = (x1 * scale, y1 * scale, x2 * scale, y2 * scale)
                self.focusing_xy.append(full_pos)
                self.focusing_each[f"Particle_{pid}"] = crop_uint8

        # resize 回全尺寸
        self.focusing = cv2.resize(focusing_small, (w_full, h_full), interpolation=cv2.INTER_LINEAR)

        elapsed = time.time() - start_time
        self._hologram.status_msg = (
            f"Multi PCHIP Focus Done. "
            f"{len(particle_finders)} particles, "
            f"Time: {elapsed:.1f}s"
        )

        return self.focusing, self.focusing_z, self.focusing_xy

    def modify_hologram_and_config(self):
        """更新 hologram 对象"""
        self._hologram.focusing = self.focusing
        self._hologram.focusing_z = self.focusing_z
        self._hologram.focusing_xy = self.focusing_xy
        self._hologram.focusing_each = self.focusing_each


# ====================================================================
# GPU 加速版本
# ====================================================================
class MultiFocusPCHIP_GPU(MultiFocusPCHIP):
    """
    多颗粒 PCHIP 自聚焦（GPU 加速版）
    完全使用 PyTorch 进行批量角谱重建，避免 CuPy 与 PyTorch 的 CUDA 上下文冲突。
    """

    def _batch_reconstruct(self, z_values: List[float]) -> Dict[float, np.ndarray]:
        """
        PyTorch 实现的批量角谱重建。
        输入：需要重建的 z 值列表
        返回：{z: 振幅图像(缩放尺寸, numpy)}
        """
        # 频谱裁剪（与原始逻辑一致）
        scale = self.rcf_scale
        M_full, N_full = self._hologram.spectrum.shape
        M_crop = M_full // scale
        N_crop = N_full // scale
        M_start = (M_full - M_crop) // 2
        N_start = (N_full - N_crop) // 2
        spectrum_cropped = self._hologram.spectrum[
            M_start:M_start + M_crop, N_start:N_start + N_crop
        ]

        # 转为 PyTorch tensor (complex64 保持精度，设备与模型一致)
        spectrum = torch.as_tensor(spectrum_cropped, device=self.device, dtype=torch.complex64)

        # 缩放后的像素尺寸
        pixel_size_scaled = self.pixel_size * scale
        wavelength = self.wavelength

        M, N = spectrum.shape

        # 生成频率坐标（与 CuPy 版本一致）
        fft_x = torch.fft.fftshift(torch.fft.fftfreq(N, d=pixel_size_scaled, device=self.device))
        fft_y = torch.fft.fftshift(torch.fft.fftfreq(M, d=pixel_size_scaled, device=self.device))
        fft_mesh_x, fft_mesh_y = torch.meshgrid(fft_x, fft_y, indexing='xy')
        fft_squa = fft_mesh_x ** 2 + fft_mesh_y ** 2

        # 去重并排序 z 值
        z_unique = sorted(set(z_values))
        z_tensor = torch.tensor(z_unique, device=self.device, dtype=torch.float32)

        # 传播核: H = exp(j*2π/λ*z) * exp(-j*π*λ*z*(fx^2+fy^2))
        k0 = 2 * torch.pi / wavelength
        phase1 = k0 * z_tensor[:, None, None]                     # (num_z, 1, 1)
        phase2 = -torch.pi * wavelength * z_tensor[:, None, None] * fft_squa[None, :, :]
        H = torch.exp(1j * (phase1 + phase2))

        # 批量重建
        A = H * spectrum[None, :, :]                               # (num_z, M, N)
        # 注意: PyTorch 的 ifft2 期望输入形状为 (..., M, N)，dim=(-2,-1) 默认正确
        # 先 ifftshift 再 ifft2
        A_shifted = torch.fft.ifftshift(A, dim=(-2, -1))          # 将零频移到中心
        U = torch.fft.ifft2(A_shifted, dim=(-2, -1))              # 复数场

        # 取振幅并转回 numpy（同步 GPU 操作）
        amplitude = torch.abs(U)

        # 确保所有计算完成后再取数据
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        result = {}
        for i, z in enumerate(z_unique):
            result[z] = amplitude[i].cpu().numpy()

        # 清理显存（可选）
        del spectrum, H, A, U, amplitude
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return result