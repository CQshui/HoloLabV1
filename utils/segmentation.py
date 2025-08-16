from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import os
import cv2
import numpy as np

try:
    import torch
    import segmentation_models_pytorch as smp

except Exception as e:
    print(f"[Module] {e}")

class Segmentation_LZM():
    def __init__(self, hologram=None, config=None):
        self.focusing       = hologram.focusing
        self.segmentation   = None

        self.gray_threshold = config.segmentation['gray_thresh'] # config.focusing['method']
        self.block_size     = config.segmentation['block_size']

    def run(self):
        _, binary_img = cv2.threshold(self.focusing, self.gray_threshold, 255, cv2.THRESH_BINARY_INV)

        self.segmentation = binary_img

        return self.segmentation

    def Segmentation_Global_Threshold_CPU(self):
        a = 1
    def Segmentation_Adaptive_Threshold_CPU(self):
        a = 1
    def Segmentation_Machine_Learning_CPU(self):
        a = 1
    def Segmentation_Global_Threshold_GPU(self):
        a = 1
    def Segmentation_Adaptive_Threshold_GPU(self):
        a = 1
    def Segmentation_Machine_Learning_GPU(self):
        a = 1

'''LuRou@2025.06.12'''
class ModelManager:
    _instance = None

    def __new__(cls, backbone='efficientnet-b3', num_classes=1, activation="sigmoid", model_path=None, device='cpu'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if model_path is None:
                raise ValueError("model_path must be provided to ModelManager")
            cls._instance._init_model(backbone, num_classes, activation, model_path, device)
        return cls._instance

    def _init_model(self, backbone, num_classes, activation, model_path, device):
        self.device = torch.device(device)
        self.backbone = backbone

        # 预处理函数
        self.preprocess_fn = smp.encoders.get_preprocessing_fn(backbone, pretrained='imagenet')

        # 初始化模型
        self.model = smp.Unet(
            encoder_name=backbone,
            classes=num_classes,
            activation=activation,
            encoder_weights=None  # None / 'imagenet'
        ).to(self.device)

        # 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        # print(f"成功从 {model_path} 加载模型权重")

        self.model.eval()

    def predict(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """接受已经预处理并在 device 上的 Tensor，返回网络输出"""
        with torch.no_grad():
            return self.model(img_tensor)
def create_model(model_path, device):
    # 创建模型实例
    return ModelManager(model_path=model_path, device=device)

class Segmentation:
    def __init__(self, hologram, config):
        self.hologram_instance  = hologram    # 保存 hologram 实例以便后续赋值
        self.config_instance    = config      # 保存config实例以备用

        self.segmentation_type  = config.segmentation['type']  # 'holo' or 'polar'
        self.model_path         = config.segmentation['model_path']
        self.method             = config.segmentation['method']
        self.device             = config.segmentation['device']
        self.pixel_size         = config.image_info['pixel_size']

        self.gray_threshold     = config.segmentation['gray_thresh'] # config.focusing['method']
        self.block_size         = config.segmentation['block_size']

        self.cpu_num            = config.segmentation['cpu_num']  # note 多线程
        self.gpu_num            = config.segmentation['gpu_num']  # note 多GPU

        self.model_manager      = None

        if 1:
            self.save_action    = config.save_and_load['save_segmentation']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

        # 根据 segmentation_type 选择用于分割的输入图像
        if   self.segmentation_type == 'holo':
            self.focus = hologram.focusing
            print("分割类型: 'holo'. 使用 hologram.focusing 作为分割输入。")
        elif self.segmentation_type == 'polar':
            # 假设 Polarization 模块已运行，hologram.hologram_p000 (I0) 已被填充
            # 或者使用 S0 (hologram.S0) 如果它更合适且已填充
            self.focus = hologram.hologram_p000
            if self.focus is None or self.focus.size == 0:
                print("分割类型为 'polar'，但 hologram.hologram_p000 (I0) 无效或未被偏振模块填充。")
            print(
                f"分割类型: 'polar'. 使用 hologram.hologram_p000 (I0) 作为分割输入。形状: {self.focus.shape}")
        else:
            hologram.status_msg = f"错误: 未知分割类型 '{self.segmentation_type}'"
            raise ValueError(f"未知的分割类型: {self.segmentation_type}。请选择 'holo' 或 'polar'。")

        if not isinstance(self.focus, np.ndarray) or self.focus.size == 0:
            hologram.status_msg = f"严重错误: 分割输入准备失败。"
            raise ValueError("用于分割的输入图像无效或为空。")

        default_empty_image_shape = (50, 50)  # 或者从 config 获取默认图像大小
        if hasattr(hologram.hologram_raw, 'shape') and hologram.hologram_raw.size > 0:  # 用原始全息图的尺寸作为空图像的默认尺寸
            default_empty_image_shape = hologram.hologram_raw.shape[:2]

        # 分割结果的内部存储
        self.segmentation_result        = hologram._make_empty_image(shape=default_empty_image_shape)  # 灰色空图
        self.binary_particles_map       = {}    # { "idx_coords": cropped_binary_from_segmentation_result }
        self.source_image_particles_map = {}    # { "idx_coords": cropped_from_focus }

        # 如果是 polar 类型，还需要为 AoP 和 DoLP 的颗粒图准备存储
        if self.segmentation_type == 'polar':
            self.aop_particles_map = {}   # { "idx_coords": cropped_AoP_array }
            self.dolp_particles_map = {}  # { "idx_coords": cropped_DoLP_array }

    def run(self):
        """根据方法名称执行不同函数"""
        if   self.method == 'Global_Threshold':
            self.segmentation_global_threshold()

        elif self.method == 'Adaptive_Threshold':
            self.segmentation_adaptive_threshold()

        elif self.method == 'Deep_Learning':
            if not self.model_path or not os.path.exists(self.model_path):
                raise ValueError("模型路径无效: {self.model_path}")
            self.model_manager = create_model(self.model_path, self.device)

            if   self.device == 'cpu':
                self.segmentation_deep_learning_cpu()
            elif self.device == 'cuda':
                self.segmentation_deep_learning_gpu()

        else:
            hologram.status_msg = f"不支持的分割方法: {self.method}"
            raise NotImplementedError(f"不支持的分割方法: {self.method}")

        self.modify_hologram_and_config()
        self.save_to_file()

    def modify_hologram_and_config(self):

        if self.hologram_instance is not None:
            self.hologram_instance.segmentation = self.segmentation_result
            self.hologram_instance.segmentation_each = self.binary_particles_map

            # focusing_each 将存储基于分割输入源（可能是 hologram.focusing 或 hologram.hologram_p000）的颗粒图
            self.hologram_instance.focusing_each = self.source_image_particles_map

            # 如果是偏振类型，额外处理和赋值 AoP 和 DoLP 的颗粒图
            if self.segmentation_type == 'polar':
                self._extract_polar_parameter_particles()  # 新增方法
                self.hologram_instance.aop_each = self.aop_particles_map
                self.hologram_instance.dolp_each = self.dolp_particles_map

            self.hologram_instance.status_msg = 'Segmentation Completed.'
    def save_to_file(self):
        """
        保存分割结果:
        - 完整分割掩码 (self.segmentation_result)
        - 从 self.binary_particles_map 提取的单颗粒二值化图像
        - 从 self.source_image_particles_map 提取的单颗粒原始输入图像 (可能是I0或原始聚焦图)
        - 如果是 'polar' 类型，额外保存从 self.aop_particles_map 和 self.dolp_particles_map 提取的颗粒图
        """
        # note 获取原图名称作为前缀（例如 inline_single_particle_Z0.030）  原始全息图or聚焦全息图？

        if not self.save_action:
            return

        if self.creat_sub_dir:
            mask_dir            = os.path.join(self.save_path, self.image_name, 'Segmentation_Mask')
            single_binary_dir   = os.path.join(self.save_path, self.image_name, 'Segmentation_Single_Binary')
            single_original_dir = os.path.join(self.save_path, self.image_name, 'Segmentation_Single_Origin')

            if self.segmentation_type == 'polar':
                dir_particles_aop   = os.path.join(self.save_path, self.image_name, 'aop_particles')
                dir_particles_dolp  = os.path.join(self.save_path, self.image_name, 'dolp_particles')
        else:
            mask_dir            = os.path.join(self.save_path, 'Segmentation_Mask')
            single_binary_dir   = os.path.join(self.save_path, 'Segmentation_Single_Binary')
            single_original_dir = os.path.join(self.save_path, 'Segmentation_Single_Origin')

            if self.segmentation_type == 'polar':
                dir_particles_aop   = os.path.join(self.save_path, 'aop_particles')
                dir_particles_dolp  = os.path.join(self.save_path, 'dolp_particles')

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(single_binary_dir, exist_ok=True)
        os.makedirs(single_original_dir, exist_ok=True)
        if self.segmentation_type == 'polar':
            os.makedirs(dir_particles_aop, exist_ok=True)
            os.makedirs(dir_particles_dolp, exist_ok=True)

        # 保存二值掩膜图（保持原图大小）
        mask_filename = os.path.join(mask_dir, f"{self.image_name}_mask.png")
        cv2.imwrite(mask_filename, self.segmentation_result)

        # 2. 保存从 self.segmentation_each 提取的单颗粒二值化图像
        for key_string, cropped_binary_image in self.binary_particles_map.items():
            # 为文件名创建一个安全版本 (替换特殊字符)
            safe_filename_key = key_string.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')
            particle_name = f"{safe_filename_key}_binary.png"
            particle_path = os.path.join(single_binary_dir, particle_name)
            cv2.imwrite(particle_path, cropped_binary_image)

        # 3. 保存从 self.focusing_each 提取的单颗粒原始图像
        for key_string, cropped_original_image in self.source_image_particles_map.items():
            safe_filename_key = key_string.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')
            particle_name = f"{safe_filename_key}_original.png"
            particle_path = os.path.join(single_original_dir, particle_name)
            cv2.imwrite(particle_path, cropped_original_image)

        # 4. 如果是 'polar' 类型，额外保存 AoP 和 DoLP 颗粒图
        if self.segmentation_type == 'polar':
            for key_string, img_data in self.aop_particles_map.items():
                safe_fn_key = key_string.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')
                fn = f"{safe_fn_key}_aop.png"
                # AoP图像可能是float，保存时需要注意。如果已转为uint8可视化版本，则直接保存。
                # 如果是float，可以保存为 .tiff 或归一化后保存为 .png
                cv2.imwrite(os.path.join(dir_particles_aop, fn), img_data)  # 直接保存uint8

            for key_string, img_data in self.dolp_particles_map.items():
                safe_fn_key = key_string.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')
                fn = f"{safe_fn_key}_dolp.png"
                # DoLP 图像通常是 uint8 (0-255)
                cv2.imwrite(os.path.join(dir_particles_dolp, fn), img_data)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理：确保为3通道BGR，然后转RGB，调整大小，归一化"""
        # 如果图像为单通道，先转 BGR
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # 转为 RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        processed = self.model_manager.preprocess_fn(img_resized)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    def _postprocess(self, pred: np.ndarray, original_shape: tuple) -> np.ndarray:
        """后处理：调整尺寸并二值化"""
        mask = (pred > 0.5).astype(np.uint8) * 255
        return cv2.resize(mask, (original_shape[1], original_shape[0]))
    def _extract_single_particles(self, mask: np.ndarray, min_area_um2: float = 20.0, edge_generation: int = 5):
        """
        - 从完整二值掩码中提取单个颗粒。
        填充 self.binary_particles_map (裁剪的二值颗粒)
        填充 self.source_image_particles_map (从 self.focus 裁剪的对应原始颗粒)
        """
        self.binary_particles_map.clear()
        self.source_image_particles_map.clear()

        # 计算每像素对应的微米尺寸: pixel_size 单位是米，将它除以 unit_um(1e-6) 得到像素的微米值
        pix_um = self.pixel_size / unit_um

        # 找到所有连通域 (轮廓)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particle_idx_counter = 0
        img_height, img_width = self.focus.shape[:2]  # 从原始聚焦图获取尺寸
        for cnt in contours:
            area_px2 = cv2.contourArea(cnt)
            if area_px2 == 0:  # 跳过面积为零的轮廓
                continue

            area_um2 = area_px2 * (pix_um ** 2)
            # 面积过滤：剔除过小的噪声
            if area_um2 < min_area_um2:
                continue

            particle_idx_counter += 1

            # 计算边界框，并加上边距
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(0, x - edge_generation)
            y1 = max(0, y - edge_generation)
            x2 = min(img_width, x + w + edge_generation)
            y2 = min(img_height, y + h + edge_generation)

            if x1 >= x2 or y1 >= y2:
                continue

            # 创建键名
            key_string = f"{particle_idx_counter}_({x1},{y1},{x2},{y2})"

            # 裁剪二值化颗粒图像
            cropped_binary_particle = mask[y1:y2, x1:x2]

            # 裁剪原始聚焦图像中的颗粒
            # 确保 self.focus 是二维的灰度图或三维的彩色图，以便正确裁剪
            if self.focus.ndim == 3:
                cropped_original_particle = self.focus[y1:y2, x1:x2, :]
            else:  # ndim == 2
                cropped_original_particle = self.focus[y1:y2, x1:x2]

            self.binary_particles_map[key_string] = cropped_binary_particle
            self.source_image_particles_map[key_string] = cropped_original_particle
    def _extract_polar_parameter_particles(self):
        """
        在 'polar' 类型分割后，使用已有的颗粒坐标信息 (从 self.binary_particles_map 的键)
        从 hologram 的整图 AoP 和 DoLP 中裁剪出对应颗粒的图像。
        """
        print("为 'polar' 类型分割提取 AoP 和 DoLP 颗粒图...")
        if self.hologram_instance.AoP is None or self.hologram_instance.DoLP is None:
            print("警告: hologram 实例中整图 AoP 或 DoLP 未被填充，无法提取偏振参数颗粒图。")
            return

        full_aop_map = self.hologram_instance.AoP
        full_dolp_map = self.hologram_instance.DoLP

        # 确保 AoP/DoLP 图与分割所用图像 (如 I0) 尺寸一致
        # Polarization 模块在计算时应已保证这一点
        if full_aop_map.shape[:2] != self.focus.shape[:2] or \
                full_dolp_map.shape[:2] != self.focus.shape[:2]:
            print(f"警告: 整图 AoP/DoLP 尺寸 ({full_aop_map.shape[:2]}/{full_dolp_map.shape[:2]}) "
                  f"与分割输入图像尺寸 ({self.focus.shape[:2]}) 不匹配。裁剪可能不准确。")

        self.aop_particles_map.clear()
        self.dolp_particles_map.clear()

        for key_string in self.binary_particles_map.keys():  # 使用已有的颗粒键和坐标
            # 从 key_string "idx_(x1,y1,x2,y2)" 中解析坐标
            try:
                parts = key_string.split('_(')
                coords_str = parts[1][:-1]  # 移除末尾的 ')'
                x1, y1, x2, y2 = map(int, coords_str.split(','))
            except Exception as e:
                print(f"警告: 解析颗粒坐标键 '{key_string}' 失败: {e}。跳过此颗粒的AoP/DoLP提取。")
                continue

            # 裁剪 AoP 颗粒图
            if full_aop_map.ndim == 3:
                cropped_aop = full_aop_map[y1:y2, x1:x2, :]
            else:
                cropped_aop = full_aop_map[y1:y2, x1:x2]
            if cropped_aop.size > 0:
                self.aop_particles_map[key_string] = cropped_aop

            # 裁剪 DoLP 颗粒图
            if full_dolp_map.ndim == 3:
                cropped_dolp = full_dolp_map[y1:y2, x1:x2, :]
            else:
                cropped_dolp = full_dolp_map[y1:y2, x1:x2]
            if cropped_dolp.size > 0:
                self.dolp_particles_map[key_string] = cropped_dolp
        print("AoP 和 DoLP 颗粒图提取完成。")

    def segmentation_global_threshold(self):
        """全局阈值分割（CPU版）"""
        # 预处理：确保为灰度图
        if self.focus.ndim == 3 and self.focus.shape[2] == 3:
            gray = cv2.cvtColor(self.focus, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.focus.copy()

        '使用 otsu 算法计算全局阈值（反向二值化）'
        _, binary = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )

        '''直接全局'''
        _, binary = cv2.threshold(gray, self.gray_threshold, 255, cv2.THRESH_BINARY_INV)

        # 保存结果
        self.segmentation_result = binary  # 存储完整掩码
        self._extract_single_particles(self.segmentation_result)  # 提取颗粒
    def segmentation_adaptive_threshold(self):
        """自适应阈值分割（CPU版，基于高斯加权 + 反向二值）"""
        # 确保是灰度图
        if self.focus.ndim == 3 and self.focus.shape[2] == 3:
            gray = cv2.cvtColor(self.focus, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.focus.copy()

        # # 轻微高斯模糊，减少噪声对局部阈值计算的干扰
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 反向自适应阈值：颗粒为黑色（0），背景为白色（255）
        binary = cv2.adaptiveThreshold(
            gray,
            255,  # 最大值
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 自适应方法
            cv2.THRESH_BINARY_INV,  # 反向阈值
            blockSize=11,  # 邻域大小 (必须是奇数)
            C=2  # 从均值或加权均值中减去的常数
        )

        # 保存结果
        self.segmentation_result = binary  # 存储完整掩码
        self._extract_single_particles(self.segmentation_result)  # 提取颗粒
    def segmentation_deep_learning_cpu(self):
        """基于 CPU 的机器学习分割流程"""
        torch.set_num_threads(self.cpu_num)
        input_tensor = self._preprocess(self.focus).cpu()
        pred_tensor = self.model_manager.predict(input_tensor)
        pred_numpy = pred_tensor.squeeze().cpu().numpy()

        self.segmentation_result = self._postprocess(pred_numpy, self.focus.shape[:2])
        # 保存结果
        self._extract_single_particles(self.segmentation_result)  # 提取颗粒
    def segmentation_deep_learning_gpu(self):
        """DL分割流程"""
        if not torch.cuda.is_available():
            print("CUDA 不可用。深度学习将回退到 CPU 执行。")
            self.segmentation_deep_learning_cpu()  # Fallback to CPU
            return

        input_tensor = self._preprocess(self.focus)
        pred_tensor = self.model_manager.predict(input_tensor)
        pred_numpy = pred_tensor.squeeze().cpu().numpy()

        self.segmentation_result = self._postprocess(pred_numpy, self.focus.shape[:2])
        self._extract_single_particles(self.segmentation_result)

if __name__ == '__main__':
    print('Utils Segmentation Module', end='\n\n')

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