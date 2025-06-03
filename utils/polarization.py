from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

'''
1、输入：全息图 中的 重建图像簇（重建完成之后的）self.reconstruction
2、输出：根据实际情况确定即可，如：self.AOP、self.DOLP

- 上述这些参数，在Hologram这个类中已经定义，可以直接返回给它的实体
- 先看看Hologram这个类中有哪些量

'''

class Polarization():
    def __init__(self, hologram=None, config=None):

        self.hologram      = hologram.hologram
        self.hologram_p000 = hologram.hologram_p000
        self.hologram_p045 = hologram.hologram_p045
        self.hologram_p090 = hologram.hologram_p090
        self.hologram_p135 = hologram.hologram_p135

        self.AOP           = hologram.AOP
        self.DOLP          = hologram.DOLP

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