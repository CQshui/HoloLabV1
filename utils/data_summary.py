from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    # 'font.sans-serif': ['SimHei', 'Arial'],  # 优先使用 SimHei（黑体），失败时回退到 Arial
    'font.serif': ['Times New Roman'],

    # 'axes.unicode_minus': False,  # 解决负号显示问题

    # 'font.size': 12,
    'font.weight': 'normal',
    'figure.titlesize': 14,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.direction': 'in',  # X 轴刻度线向内
    'ytick.direction': 'in',  # Y 轴刻度线向内
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    # 新增的线宽设置
    'axes.linewidth': 1.0,  # 坐标轴线宽
    'xtick.major.width': 1.0,  # X 轴主刻度线宽
    'ytick.major.width': 1.0,  # Y 轴主刻度线宽
    'xtick.minor.width': 1.0,  # X 轴次刻度线宽（如果有）
    'ytick.minor.width': 1.0  # Y 轴次刻度线宽（如果有）
})

class DataSummary:
    def __init__(self, hologram=None, config=None):
        a = 1

        self.hologram = hologram

        self.show_size_distribution  = config.data_summary['show_size_distribution']
        self.show_concentration_time = config.data_summary['show_concentration_time']
        self.show_R50_distribution   = config.data_summary['show_R50_distribution']
        self.show_R90_distribution   = config.data_summary['show_R90_distribution']
        self.show_R200_distribution  = config.data_summary['show_R200_distribution']

        self.data_fig = {}

    def run(self):

        fig_key, fig = self.Data_Figure_Example()
        self.data_fig[fig_key] = fig

        return self.data_fig

    def Data_Figure_Example(self):

        fig1 = Figure()
        ax = fig1.add_subplot(111)
        ax.plot([0, 1, 2], [2, 1, 3])
        ax.set_title("Example Plot")

        fig2 = Figure()
        ax = fig2.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_title("Example Plot")

        fig_key = 'Size Distribution'
        return fig_key, fig1

    def Particle_Size_Information(self):
        a = 1

    def Particle_Size_Distribution(self):
        a = 1

    def Particle_Size_Concentration(self):
        a = 1

    def Particle_Size_Category(self):
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