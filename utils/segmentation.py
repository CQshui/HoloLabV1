from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

'''
1、输入：全息图 中的 聚焦完成的图 self.focusing、或者可能是重建完成的图簇 self.reconstruction
2、输出：分割的Mask图 self.segmentation，以及进一步的self.segmentation_each

- 上述这些参数，在Hologram这个类中已经定义，可以直接返回给它的实体
- 先看看Hologram这个类中有哪些量

'''

class Segmentation():
    def __init__(self, hologram=None, config=None):
        a = 1

    def run(self):
        class_name = self.__class__.__name__
        print(f"Running operation in class: {class_name}")

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