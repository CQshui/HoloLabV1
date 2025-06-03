from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

'''
1、输入：全息图 中的 分割完成的图 self.segmentation
2、输出：识别图 
    在一张图中 self.identification
    截取出来的 self.identification_each，这是个list，里面包含了每个颗粒的识别图

- 上述这些参数，在Hologram这个类中已经定义，可以直接返回给它的实体
- 先看看Hologram这个类中有哪些量

'''

class Identification():
    def __init__(self, hologram, config):
        a = 1

    def run(self):
        class_name = self.__class__.__name__
        print(f"Running operation in class: {class_name}")

    def Identification_Unet(self):
        a = 1
    def Identification_Others(self):
        a = 1

if 0:
    class Identification():
        def __init__(self, hologram=None, config=None):
            self.hologram = hologram  # 保存 hologram 引用

            # 修改为直接使用NumPy数组
            self.aop = hologram.aop  # {filename: aop_array}
            self.dolp = hologram.dolp  # {filename: dolp_array}
            self.image_dict = hologram.focusing_each  # {filename: amp_array}

            self.method = config.identification['method']
            self.type = config.identification['type']
            self.classes_names = config.identification['type_dict'][self.type]
            self.model_path = config.identification['model_path']
            self.model_name = config.identification['model_name']

        def run(self):
            if self.method == 'onlyholo':
                return self.Identification_onlyholo()
            elif self.method == 'polar':
                return self.Identification_polarization()
            else:
                raise ValueError(f"Unknown method: {self.method}")

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
                    print(f"{filename} → {pred} "
                          f"(Top1: {top2.values[0]:.3f}, Top2: {top2.values[1]:.3f})")

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
                    print(f"{filename} → {pred} "
                          f"(Top1: {top2.values[0]:.3f}, Top2: {top2.values[1]:.3f})")

                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")

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