from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

'''
1、输入：波前
2、输出：相位相关的图 
        self.phase
        self.phase_unwrapped
        self.phase_compensated
        self.phase_compensation_mat
        self.phase_corrected
    
- 上述这些参数，在Hologram这个类中已经定义，可以直接返回给它的实体
- 先看看Hologram这个类中有哪些量

'''

class Phase():
    def __init__(self, hologram, config):
        a = 1

    def run(self):
        class_name = self.__class__.__name__
        print(f"Running operation in class: {class_name}")

    def Phase_Unwrapping(self):
        a = 1
    def Phase_Compensation_Zernike(self, modes):
        a = 1
    def Phase_Compensation_Machine_Learning(self, modes):
        a = 1

if __name__ == '__main__':
    print('Utils Phase Analysis Module', end='\n\n')

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