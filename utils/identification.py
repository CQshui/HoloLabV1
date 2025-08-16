import os
import os.path as osp
import shutil
from PIL import Image
from typing import Optional, Callable
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print(f"[Module] {e}")

from utils.utils_for_identify.multi_Input import get_torch_transforms_1channel
from utils.utils_for_identify.multiinput_polar import get_torch_transforms_3channel
from utils.utils_for_identify.train import SELFMODEL1
from utils.utils_for_identify.train_polar import SELFMODEL2

from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

class Identification_LZM():
    def __init__(self, hologram, config):
        a = 1

    def run(self):
        class_name = self.__class__.__name__
        print(f"Running operation in class: {class_name}")

    def Identification_Unet(self):
        a = 1
    def Identification_Others(self):
        a = 1

class Identification_:

    def __init__(self, hologram=None,config=None):
        """
        :param hologram: Hologram 对象，包含 focusing_each 字典
        :param config: 配置对象
        """
        self.hologram   = hologram  # 保存 hologram 引用

        # 修改为直接使用NumPy数组
        self.aop        = hologram.AoP      # {filename: aop_array}
        self.dolp       = hologram.DoLP     # {filename: dolp_array}
        self.image_dict = hologram.focusing_each  # {filename: amp_array}

        self.method         = config.identification['method']
        self.type           = config.identification['type']
        self.classes_names  = config.identification['type_dict'][self.type]
        self.model_path     = config.identification['model_path']
        self.model_name     = config.identification['model_name']

    def run(self):
        """
        调度执行不同的处理方法
        """
        if self.method == 'onlyholo':
            return self.Identification_onlyholo()
        elif self.method == 'polar':
            return self.Identification_polarization()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.hologram.status_msg = 'Identification Finished.'

    def Identification_onlyholo(self):
        """
        仅使用全息图像（单通道振幅）进行分类处理

        处理流程：
        1. 加载预训练模型
        2. 对每个全息图像进行预处理
        3. 执行推理并获取分类结果
        4. 将结果保存到hologram对象中

        注意：
        - 直接操作内存中的numpy数组，避免磁盘IO
        - 处理结果保存格式：{filename_pred: amplitude_array}
        """
        # 1. 模型初始化
        model_path = osp.join(self.model_path, self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1.1 创建模型实例（单通道输入）
        self.model = SELFMODEL1(
            model_name="resnet50d",
            out_features=len(self.classes_names),
            pretrained=False
        )

        # 1.2 加载预训练权重
        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()

        # 1.3 获取单通道图像预处理方法
        self.transforms = get_torch_transforms_1channel(img_size=224)['val']

        self.hologram.identification_each = {}

        # 2. 检查输入数据
        total = len(self.image_dict)
        if total == 0:
            print("警告：未找到待分类图像。")
            return

        # 3. 批量处理全息图像
        for filename, img_array in self.image_dict.items():
            try:
                # 3.1 数据验证（确保是2D numpy array）
                if len(img_array.shape) != 2:
                    raise ValueError(f"输入应为2D数组，实际得到{img_array.shape}")

                # 3.2 添加通道维度 (H, W) -> (H, W, 1)
                arr = img_array[..., None]

                # 3.3 预处理并转换为tensor
                tensor = self.transforms(arr).unsqueeze(0).to(self.device)

                # 3.4 执行推理
                with torch.no_grad():
                    out = self.model(tensor)

                # 3.5 获取预测结果
                label_id = torch.argmax(out, dim=1).item()
                pred = self.classes_names[label_id]

                # 3.6 创建新键名：filename_pred
                new_key = f"{filename}_{pred}"

                # 3.7 保存结果（原始2D数组）
                self.hologram.identification_each[new_key] = img_array

                # 4. 打印分类结果（调试信息）
                probs = nn.Softmax(dim=1)(out)[0]
                top2 = torch.topk(probs, 2)
                # print(f"{filename} → {pred} "
                #       f"(Top1: {top2.values[0]:.3f}, Top2: {top2.values[1]:.3f})")

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

    def Identification_polarization(self):
        """
        使用偏振三通道图像（振幅+AoP+DoLP）进行分类处理

        处理流程：
        1. 加载预训练模型
        2. 合并三通道数据（振幅、AoP、DoLP）
        3. 执行推理并获取分类结果
        4. 将结果保存到hologram对象中

        注意：
        - 直接从内存获取三通道数据，避免磁盘IO
        - 处理结果保存格式：{filename_类型: amplitude_array}
        """
        # 1. 模型初始化
        model_path = osp.join(self.model_path, self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1.1 创建模型实例（三通道输入）
        self.model = SELFMODEL2(
            model_name="resnet50d",
            out_features=len(self.classes_names),
            pretrained=False
        )

        # 1.2 加载预训练权重
        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()

        # 1.3 获取三通道图像预处理方法
        self.transforms = get_torch_transforms_3channel(img_size=224)['val']

        # 2. 清空原有输出（避免重复）
        self.hologram.identification_each = {}

        # 3. 批量处理偏振图像
        for filename, amp_array in self.image_dict.items():
            try:
                # 3.1 从内存获取三通道数据
                aop_array = self.aop[filename]
                dolp_array = self.dolp[filename]

                # 3.2 合并通道 [H,W,3]
                merged_img = np.stack([amp_array, aop_array, dolp_array], axis=2)

                # 3.3 预处理并转换为tensor
                tensor = self.transforms(merged_img).unsqueeze(0).to(self.device)

                # 3.4 执行推理
                with torch.no_grad():
                    out = self.model(tensor)

                # 3.5 获取预测结果
                label_id = torch.argmax(out, dim=1).item()
                pred = self.classes_names[label_id]

                # 3.6 创建输出键名
                output_key = f"{filename}_{pred}"

                # 3.7 保存结果（仅振幅通道）
                self.hologram.identification_each[output_key] = amp_array.copy()

                # 4. 打印分类结果（调试信息）
                probs = nn.Softmax(dim=1)(out)[0]
                top2 = torch.topk(probs, 2)
                # print(f"{filename} → {pred} "
                #       f"(Top1: {top2.values[0]:.3f}, Top2: {top2.values[1]:.3f})")

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

'''LiCH@2025.06.12'''
class Identification:

    def __init__(self, hologram = None, config = None):
        self._hologram      = hologram  # 保存 hologram 引用
        self._config        = config    # 保存 config 引用

        # 修改为直接使用NumPy数组
        self.AoP            = self._hologram.AoP  # {filename: aop_array}
        self.DoLP           = self._hologram.DoLP  # {filename: dolp_array}
        self.image_dict     = self._hologram.focusing_each  # {filename: amp_array}

        self.method         = self._config.identification['method']
        self.type           = self._config.identification['type']
        self.classes_names  = self._config.identification['type_dict'][self.type]
        self.model_path     = self._config.identification['model_path']
        self.model_name     = self._config.identification['model_name']

        if 1:
            self.save_action    = config.save_and_load['save_identification']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

        self.identification_each    = {}    # 赋值变量
        self.identification_judge   = True

    def run(self):
        if   self.method == 'onlyholo':
            # 调用处理方法，该方法内部会设置 identification_judge
            self.Identification_onlyholo()

        elif self.method == 'polar':
            # 调用处理方法，该方法内部会设置 identification_judge
            self.Identification_polarization()

        else:
            # 如果方法名不匹配，直接标记为错误
            self.identification_judge = False
            print(f"警告：未知的识别方法 '{self.method}'。")

        self.modify_hologram_and_config()
        self.save_to_file()

    def modify_hologram_and_config(self):
        """
        根据识别结果修改 hologram 对象的状态和数据。
        """
        if self.identification_judge == True:
            self._hologram.identification_each = self.identification_each
            self._hologram.status_msg = 'Identification Completed'

        else:
            self._hologram.identification_each = {}
            self._hologram.status_msg = 'Identification Wrong'

    def save_to_file(self):
        if self.creat_sub_dir:
            save_path = os.path.join(self.save_path, self.image_name)
            save_path = os.path.join(save_path, 'Identification')
        else:
            save_path = os.path.join(self.save_path, 'Identification')

        if self.save_action:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            identification = self.identification_each
            # for _key, _value in identification.items():
            #     abs_v = np.abs(_value.copy())
            #     normalized = (abs_v - np.min(abs_v)) / (np.max(abs_v) - np.min(abs_v) + 1e-10)
            #     identification[_key] = (normalized * 255).astype(np.uint8)  # 转为 [0, 255] 的 uint8

            for _name, image in identification.items():
                save_name       = f"{_name}.png"
                save_path_file  = os.path.join(save_path, save_name)
                cv2.imwrite(save_path_file, image)

    def Identification_onlyholo(self):
        """
        仅使用全息图像（单通道振幅）进行分类处理

        处理流程：
        1. 加载预训练模型
        2. 对每个全息图像进行预处理
        3. 执行推理并获取分类结果
        4. 将结果保存到hologram对象中

        注意：
        - 直接操作内存中的numpy数组，避免磁盘IO
        - 处理结果保存格式：{filename_pred: amplitude_array}
        - 任何处理错误都会将 self.identification_judge 置为 False 并停止该方法。
        """
        try:
            # 1. 模型初始化
            model_path = osp.join(self.model_path, self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 1.1 创建模型实例（单通道输入）
            self.model = SELFMODEL1(
                model_name="resnet50d",
                out_features=len(self.classes_names),
                pretrained=False
            )

            # 1.2 加载预训练权重
            weights = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(weights)
            self.model.to(self.device).eval()

            # 1.3 获取单通道图像预处理方法
            self.transforms = get_torch_transforms_1channel(img_size=224)['val']

            # 2. 检查输入数据
            total = len(self.image_dict)
            if total == 0:
                print("警告：未找到待分类图像。")
                self.identification_judge = False  # 没有图像也算作一种“不成功”
                return  # 提前退出

            # 3. 批量处理全息图像
            for filename, img_array in self.image_dict.items():
                try:
                    # 3.1 数据验证（确保是2D numpy array）
                    if len(img_array.shape) != 2:
                        print(f"处理 {filename} 时出错: 输入应为2D数组，实际得到{img_array.shape}。跳过此文件。")
                        self.identification_judge = False  # 即使跳过，也标记整体失败
                        continue  # 跳过当前文件，继续下一个

                    # 3.2 添加通道维度 (H, W) -> (H, W, 1)
                    arr = img_array[..., None]

                    # 3.3 预处理并转换为tensor
                    tensor = self.transforms(arr).unsqueeze(0).to(self.device)

                    # 3.4 执行推理
                    with torch.no_grad():
                        out = self.model(tensor)

                    # 3.5 获取预测结果
                    label_id = torch.argmax(out, dim=1).item()
                    pred = self.classes_names[label_id]

                    # 3.6 创建新键名：filename_pred
                    new_key = f"{filename}_{pred}"

                    # 3.7 保存结果（原始2D数组）
                    self.identification_each[new_key] = img_array

                    # 4. 打印分类结果（调试信息）
                    probs = nn.Softmax(dim=1)(out)[0]
                    top2 = torch.topk(probs, 2)
                    print(f"{filename} → {pred} "
                          f"(Top1: {top2.values[0]:.3f}, Top2: {top2.values[1]:.3f})")

                except Exception as e:
                    print(f"处理 {filename} 时发生内部错误: {str(e)}。跳过此文件。")
                    self.identification_judge = False  # 标记整体失败

        except Exception as e:
            # 捕获 Identification_onlyholo 方法中的任何全局错误（例如模型加载失败）
            print(f"识别方法 Identification_onlyholo 发生全局错误: {str(e)}")
            self.identification_judge = False

    def Identification_polarization(self):
        """
        使用偏振三通道图像（振幅+AoP+DoLP）进行分类处理

        处理流程：
        1. 加载预训练模型
        2. 合并三通道数据（振幅、AoP、DoLP）
        3. 执行推理并获取分类结果
        4. 将结果保存到hologram对象中

        注意：
        - 直接从内存获取三通道数据，避免磁盘IO
        - 处理结果保存格式：{filename_类型: amplitude_array}
        - 任何处理错误都会将 self.identification_judge 置为 False 并停止该方法。
        """
        try:
            # 1. 模型初始化
            model_path = osp.join(self.model_path, self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 1.1 创建模型实例（三通道输入）
            self.model = SELFMODEL2(
                model_name="resnet50d",
                out_features=len(self.classes_names),
                pretrained=False
            )

            # 1.2 加载预训练权重
            weights = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(weights)
            self.model.to(self.device).eval()

            # 1.3 获取三通道图像预处理方法
            self.transforms = get_torch_transforms_3channel(img_size=224)['val']

            # 2. 检查输入数据
            total_amp = len(self.image_dict)
            total_aop = len(self.AoP)
            total_dop = len(self.DoLP)

            if total_amp == 0 or total_aop == 0 or total_dop == 0:
                print("警告：amp, aop, dop其中之一输入为0，未找到图像。")
                self.identification_judge = False
                return  # 提前退出


            # 3. 批量处理偏振图像
            for filename, amp_array in self.image_dict.items():
                try:
                    # 3.1 从内存获取三通道数据
                    aop_array = self.AoP[filename]
                    dolp_array = self.DoLP[filename]

                    # 3.2 合并通道 [H,W,3]
                    # 验证 shape 是否一致，防止合并失败
                    if not (amp_array.shape == aop_array.shape == dolp_array.shape and len(amp_array.shape) == 2):
                        print(f"处理 {filename} 时出错: 振幅、AoP或DoLP数组形状不匹配或不是2D。跳过此文件。")
                        self.identification_judge = False
                        continue  # 跳过当前文件

                    merged_img = np.stack([amp_array, aop_array, dolp_array], axis=2)

                    # 3.3 预处理并转换为tensor
                    tensor = self.transforms(merged_img).unsqueeze(0).to(self.device)

                    # 3.4 执行推理
                    with torch.no_grad():
                        out = self.model(tensor)

                    # 3.5 获取预测结果
                    label_id = torch.argmax(out, dim=1).item()
                    pred = self.classes_names[label_id]

                    # 3.6 创建输出键名
                    output_key = f"{filename}_{pred}"

                    # 3.7 保存结果（仅振幅通道）
                    self.identification_each[output_key] = amp_array.copy()

                    # 4. 打印分类结果（调试信息）
                    probs = nn.Softmax(dim=1)(out)[0]
                    top2 = torch.topk(probs, 2)
                    print(f"{filename} → {pred} "
                          f"(Top1: {top2.values[0]:.3f}, Top2: {top2.values[1]:.3f})")

                except Exception as e:
                    print(f"处理 {filename} 时发生内部错误: {str(e)}。跳过此文件。")
                    self.identification_judge = False  # 标记整体失败
                    continue  # 跳过当前文件

        except Exception as e:
            # 捕获 Identification_polarization 方法中的任何全局错误（例如模型加载失败）
            print(f"识别方法 Identification_polarization 发生全局错误: {str(e)}")
            self.identification_judge = False

if __name__ == '__main__':
    print('Utils Identification Module', end='\n\n')

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