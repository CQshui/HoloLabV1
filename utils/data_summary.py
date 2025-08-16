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

import io
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Callable, List, Dict, Any, Tuple
from PIL import Image

class DataSummary_LZM:
    def __init__(self, hologram=None, config=None):
        self.hologram = hologram

        self.show_size_distribution  = config.data_summary['show_size_distribution']
        self.show_concentration_time = config.data_summary['show_concentration_time']
        self.show_R50_distribution   = config.data_summary['show_R50_distribution']
        self.show_R90_distribution   = config.data_summary['show_R90_distribution']
        self.show_R200_distribution  = config.data_summary['show_R200_distribution']

        self.figure_diameter        = None
        self.figure_classification  = None

    def run(self):
        self.Data_Figure_Example()
        self.modify_hologram_and_config()

    def modify_hologram_and_config(self):
        self.hologram.figure_diameter       = self.figure_diameter
        self.hologram.figure_classification = self.figure_classification

    def Data_Figure_Example(self):

        fig1 = Figure()
        ax = fig1.add_subplot(111)
        ax.plot([0, 1, 2], [2, 1, 3])
        ax.set_title("Example Plot")

        fig2 = Figure()
        ax = fig2.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_title("Example Plot")

        self.figure_diameter        = fig1
        self.figure_classification  = fig2

class DataSummary:
    def __init__(self, hologram=None, config=None ):

        self._hologram               = hologram
        self._config                 = config
        self.all_data                = []
        self.data_path               = self._config.save_and_load['data_save_path']
        self.method                  = self._config.data_summary['method']
        self.segmented_images        = self._hologram.segmentation_each
        self.identification_images   = self._hologram.identification_each

        # 从hologram实例中获取参数
        self.pixel_size              = self._config.image_info['pixel_size']
        self.image_dimensions        = (self._config.image_info['pixel_num_x'], self._config.image_info['pixel_num_y'])
        self.z1                      = self._config.reconstruction['z_start'] * unit_mm / unit_um
        self.z2                      = self._config.reconstruction['z_end'] * unit_mm / unit_um
        self.density                 = self._config.data_summary['density']

        self.type                    = self._config.identification['type']
        self.classes_names           = self._config.identification['type_dict'][self.type]

        self.size                    = self._config.data_summary['save_size']
        self.ssc                     = self._config.data_summary['save_concentration']
        self.category                = self._config.data_summary['save_category']

        self.particle_information    = []
        self.particle_total          = {}
        self.concentration           = 0.0
        self.particle_identification = {}
        self.figure_diameter         = None
        self.figure_classification   = None
        self.data_summary_judge      = True

        if 1:
            self.save_action    = config.save_and_load['save_data_summary']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

    def run(self):
        # 首先分析每张图像，获取颗粒基本信息
        self.Particle_Size_Information()

        # 根据标志位执行不同的分析
        if self.ssc == True:
            self.analyze_concentration_info()

        if self.size == True:
            self.analyze_particle_statistics()

        if self.category == True:
            self.analyze_classification_info()

        self.modify_hologram_and_config()
        self.save_to_file()
    def modify_hologram_and_config(self):
        if self.data_summary_judge:
            self._hologram.particle_information     = self.particle_information
            self._hologram.particle_total           = self.particle_total
            self._hologram.concentration            = self.concentration
            self._hologram.particle_identification  = self.particle_identification
            self._hologram.figure_diameter          = self.figure_diameter
            self._hologram.figure_classification    = self.figure_classification

            # print('_hologram.particle_information_____________')
            # print(self._hologram.particle_information)
            # print('_hologram.particle_total___________________')
            # print(self._hologram.particle_total)
            # print('_hologram.concentration____________________')
            # print(self._hologram.concentration)
            # print('_hologram.particle_identification__________')
            # print(self._hologram.particle_identification)

            self._hologram.status_msg = 'Data Summary Completed'
        else:
            self._hologram.particle_information     = []
            self._hologram.particle_total           = {}
            self._hologram.concentration            = 0
            self._hologram.particle_identification  = {}

            self._hologram.status_msg = 'Data Summary Wrong'
    def save_to_file(self):
        if self.creat_sub_dir:
            save_path = os.path.join(self.save_path, self.image_name)
            save_path = os.path.join(save_path, 'Data_Summary')
        else:
            save_path = os.path.join(self.save_path, 'Data_Summary')

        if self.save_action:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path_file = os.path.join(save_path, 'Fig - Particle Diameter.png')
            self.figure_diameter.savefig(save_path_file, dpi=300)

            save_path_file = os.path.join(save_path, 'Fig - Classification.png')
            self.figure_classification.savefig(save_path_file, dpi=300)

            if 1:
                '颗粒参数'
                variables = {
                    "Number"    : self.particle_total["total_particles"],
                    "Mean (um)" : self.particle_total["mean_diameter"],
                    "D10  (um)" : self.particle_total["d10"],
                    "D50  (um)" : self.particle_total["d50"],
                    "D90  (um)" : self.particle_total["d90"],
                    # "Std_Dev"       : self.particle_total["std_dev"],
                    # "Total_images": self.particle_total["total_images"]

                    "Concentration (kg/m3)" : self.concentration,
                }
                with open(os.path.join(save_path, "stats_basic.txt"), "w") as f:
                    for name, value in variables.items():
                        f.write(f"{name}, {value}\n")  # 输出格式：变量名=值

                '颗粒分布'
                with open(os.path.join(save_path, "stats_distribution.txt"), "w") as f:
                    for _type, _data in self.particle_total['size_distribution'].items():
                        line = f"{_type}, {_data['percentage']:.2f}%, {_data['count']}\n"
                        f.write(line)

                        # if bin_data['count'] > 0:
                        #     line = f"{bin_range}, {bin_data['count']}, ({bin_data['percentage']:.1f}%)\n"
                        #     f.write(line)

                '颗粒分类'
                with open(os.path.join(save_path, "stats_indentification.txt"), "w") as f:
                    for _type, _data in self.particle_identification.items():
                        line = f"{_type}, {_data[0]:.2f}%, {_data[1]}\n"
                        f.write(line)

    def Particle_Size_Information(self) -> List[Dict[str, Any]]:
        """
        分析所有分割图像中的颗粒特征。
        修改后根据颗粒类型使用不同的密度值计算质量。
        :return: 包含所有颗粒特征的列表
        """
        self.all_data = []

        # 检查是否有分割图像
        if not self.segmented_images:
            print("No segmented images found")
            self.data_summary_judge = False
            return []

        # 创建颗粒类型到文件名的映射
        type_mapping = {}
        for ident_filename in self.identification_images.keys():
            # 从文件名中提取基本名称和类型
            # 假设格式为"序号_filename_类型"，如"001_image_gaoyingshi"
            base_name = '_'.join(ident_filename.split('_')[:-1])
            particle_type = ident_filename.split('_')[-1].lower()
            type_mapping[base_name] = particle_type

        # 分析每张图像
        for filename, image in self.segmented_images.items():
            # 使用文件名作为标识（去掉扩展名）
            image_name = os.path.splitext(filename)[0]

            if image is None:
                print(f"无效的图像数据: {filename}")
                self.data_summary_judge = False
                continue

            # 获取当前颗粒的类型
            particle_type = type_mapping.get(image_name, None)
            if particle_type is None:
                self.data_summary_judge = False
                print(f"无法确定颗粒类型: {filename}")
                continue

            # 获取对应类型的密度
            density = self.density.get(particle_type, None)
            if density is None:
                self.data_summary_judge = False
                print(f"未找到类型 {particle_type} 的密度值")
                continue

            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area == 0:
                    continue

                perimeter = cv2.arcLength(contour, True)
                diameter = np.sqrt(4 * area / np.pi)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

                hull = cv2.convexHull(contour)
                convex_area = cv2.contourArea(hull)
                solidity = area / convex_area if convex_area > 0 else 0

                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    if defects is not None:
                        convex_defect_depths = [defect[0][3] for defect in defects]
                        max_defect_depth = max(convex_defect_depths) / 256.0 if convex_defect_depths else 0
                    else:
                        max_defect_depth = 0
                else:
                    max_defect_depth = 0

                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
                else:
                    major_axis, minor_axis, aspect_ratio = 0, 0, 0

                # 计算颗粒体积 (假设颗粒为球形)
                diameter_um = diameter * self.pixel_size
                particle_volume = (4 / 3) * np.pi * (diameter_um / 2) ** 3  # μm³

                self.all_data.append({
                    "Image": image_name,
                    "Area (px)": area,
                    "Perimeter (px)": perimeter,
                    "Diameter (px)": diameter,
                    "Convex Area (px)": convex_area,
                    "Max Convex Defect Depth (px)": max_defect_depth,
                    "Major Axis (px)": major_axis,
                    "Minor Axis (px)": minor_axis,
                    "Area (um2)": area * (self.pixel_size ** 2),
                    "Perimeter (um)": perimeter * self.pixel_size,
                    "Diameter (um)": diameter_um,
                    "Circularity": circularity,
                    "Solidity": solidity,
                    "Aspect Ratio": aspect_ratio,
                    "Convex Area (um2)": convex_area * (self.pixel_size ** 2),
                    "Max Convex Defect Depth (um)": max_defect_depth * self.pixel_size,
                    "Major Axis (um)": major_axis * self.pixel_size,
                    "Minor Axis (um)": minor_axis * self.pixel_size,
                    "Particle Volume (um3)": particle_volume,
                    "Particle Mass (kg)": particle_volume * 1e-15 * density
                })

        self.particle_information = self.all_data
        return self.all_data
    def analyze_particle_statistics(self):
        if not self.all_data:
            self.data_summary_judge = False
            return

        df = pd.DataFrame(self.all_data)

        # 计算粒径分布
        bin_edges           = np.arange(0, df["Diameter (um)"].max() + 10, 10)
        df["Diameter Bin"]  = pd.cut(df["Diameter (um)"], bins=bin_edges, right=False)
        bin_counts          = df["Diameter Bin"].value_counts().sort_index()
        total_particles     = len(df)
        self._hologram.particle_num = total_particles

        bin_percentages = (bin_counts / total_particles) * 100

        # 计算关键统计数据
        stats = {
            "mean_diameter": df["Diameter (um)"].mean(),
            "d50": np.percentile(df["Diameter (um)"], 50),
            "d10": np.percentile(df["Diameter (um)"], 10),
            "d90": np.percentile(df["Diameter (um)"], 90),
            "std_dev": df["Diameter (um)"].std(),
            "total_particles": total_particles,
            "total_images": len(self.segmented_images),
            "size_distribution": {
                str(bin): {
                    "count": int(count),
                    "percentage": float(percentage)
                }
                for bin, count, percentage in zip(
                    bin_counts.index, bin_counts.values, bin_percentages.values
                )
            }
        }

        # 计算累积百分比
        cumulative_percentage = 0
        for bin_range in stats["size_distribution"]:
            current_percentage = stats["size_distribution"][bin_range]["percentage"]
            cumulative_percentage += current_percentage
            stats["size_distribution"][bin_range]["cumulative_percentage"] = cumulative_percentage

        self.particle_total = stats
        self.plot_diameter_distribution(df, stats)

    def Particle_Size_Concentration(self, df: pd.DataFrame) -> float:
        """
        计算颗粒浓度 (kg/m³)
        修改后使用不同类型颗粒的质量总和计算浓度
        :param df: 包含所有颗粒数据的 DataFrame
        :return: 浓度值 (kg/m³)
        """
        if self.image_dimensions is None or self.z1 is None or self.z2 is None:
            self.data_summary_judge = False
            return 0.0

        else:

            # 计算单张图像的体积 (μm³)
            image_width_um = self.image_dimensions[0] * self.pixel_size
            image_height_um = self.image_dimensions[1] * self.pixel_size
            image_depth_um = abs(self.z2 - self.z1)
            single_image_volume = image_width_um * image_height_um * image_depth_um  # μm³

            # 计算总图像体积 (m³)
            total_images = 1
            total_image_volume_m3 = single_image_volume * total_images * 1e-18  # 转换为m³

            # 计算所有颗粒的总质量 (kg)
            total_mass_kg = df["Particle Mass (kg)"].sum()

            # 计算浓度 (kg/m³)
            concentration = total_mass_kg / total_image_volume_m3 if total_image_volume_m3 > 0 else 0

            return concentration
    def analyze_concentration_info(self):
        if not self.all_data:
            self.data_summary_judge = False
            return

        df = pd.DataFrame(self.all_data)

        try:
            self.concentration = float(self.Particle_Size_Concentration(df))
        except Exception as e:
            self.data_summary_judge = False
            print(f"计算浓度时出错: {e}")

    def Particle_Size_Category(self) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        统计各类别颗粒的数量和占比
        :return: 包含各类别数量和占比的元组 (counts, percentages)
        """
        # 定义类别名称
        class_counts = {name: 0 for name in self.classes_names}
        total = 0

        # 遍历字典中的每个键值对
        for filename, array in self.identification_images.items():
            # 从文件名中提取类别信息
            # 假设文件名格式为"序号_filename_类型"，如"001_image_gaoyingshi"
            parts = filename.split('_')
            if len(parts) >= 3:
                class_name = parts[-1].lower()

                # 统计有效类别
                if class_name in class_counts:
                    class_counts[class_name] += 1
                    total += 1

        percentages = {k: (v / total * 100 if total > 0 else 0) for k, v in class_counts.items()}

        return class_counts, percentages
    def analyze_classification_info(self):
        """
        分析颗粒分类信息并保存到particle_identification
        """
        counts, percentages = self.Particle_Size_Category()
        class_names = list(percentages.keys())
        percents = list(percentages.values())
        counts = list(counts.values())

        # 将统计结果存入hologram对象
        self.particle_identification = {
            class_name: [percent, count]
            for class_name, percent, count in zip(class_names, percents, counts)
        }

        self.plot_classification_results()

    ''' 柱状图形式'''
    ''' 
        但是不兼容主函数中Data_Viewer，无法调用显示
        因为 bar() 和 sns.barplot() 返回的柱子是 Rectangle 类型的 patch，
        它们不能被添加到两个不同的 Figure 中，这是你 _copy_axes_content() 中 add_patch() 触发的错误来源。
    '''
    def plot_diameter_distribution__(self, df: pd.DataFrame, stats: dict) -> Figure:
        bin_edges = np.arange(0, df["Diameter (um)"].max() + 10, 10)
        df["Diameter Bin"] = pd.cut(df["Diameter (um)"], bins=bin_edges, right=False)
        bin_counts = df["Diameter Bin"].value_counts().sort_index()
        total_particles = len(df)
        bin_percentages = (bin_counts / total_particles) * 100

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        bars = sns.barplot(x=bin_percentages.index.astype(str), y=bin_percentages.values,
                           color="skyblue", ax=ax)

        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

        if stats:
            text_lines = [
                f'Total Particles: {stats["total_particles"]}',
                f'Mean Diameter: {stats["mean_diameter"]:.2f} μm',
                f'Median (D50): {stats["d50"]:.2f} μm',
                f'Total Images: {stats["total_images"]}'
            ]
            if self.concentration != 0.0:
                text_lines.append(f'Concentration: {self.concentration:.4f} kg/m³')

            textstr = '\n'.join(text_lines)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_xticks(np.arange(len(bin_percentages.index)))
        ax.set_xticklabels(bin_percentages.index.astype(str), rotation=45, ha="right")

        ax.set_xlabel("Particle Diameter Range (um)")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Particle Size Distribution")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # 保存
        save_path = os.path.join(self.data_path, 'Fig - Particle Size Distribution.png')
        fig.savefig(save_path, dpi=300)
        print(f"Saved Classification Analysis Figure to {save_path}")
        self.figure_diameter = fig
        return fig
    def plot_classification_results__(self) -> Figure:
        if not self.particle_identification:
            return None

        class_info = self.particle_identification
        class_names = list(class_info.keys())
        percents = [info[0] for info in class_info.values()]
        counts = [info[1] for info in class_info.values()]

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        bars = ax.bar(class_names, percents, color='skyblue')

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{counts[i]}\n({height:.1f}%)',
                    ha='center', va='bottom')

        title = 'Particle Classification Distribution'
        if self.method == 'polar':
            title += ' (Amplitude Images)'

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Particle Type', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, max(percents) * 1.2 if percents else 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")

        save_path = os.path.join(self.data_path, 'Fig - Particle Classification Distribution.png')
        fig.savefig(save_path, dpi=300)
        print(f"Saved Classification Analysis Figure to {save_path}")

        self.figure_classification = fig
        return fig

    '''曲线图形式'''
    def plot_diameter_distribution(self, df: pd.DataFrame, stats: dict):
        bin_edges = np.arange(0, df["Diameter (um)"].max() + 10, 10)
        df["Diameter Bin"] = pd.cut(df["Diameter (um)"], bins=bin_edges, right=False)
        bin_counts = df["Diameter Bin"].value_counts().sort_index()
        total_particles = len(df)
        bin_percentages = (bin_counts / total_particles) * 100

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # 生成折线图代替柱状图
        x = np.arange(len(bin_percentages))
        y = bin_percentages.values

        ax.plot(x, y, marker='o', linestyle='-', color='skyblue', label="Percentage")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_percentages.index.astype(str), rotation=45, ha="right")

        # 添加百分比标签
        for i, val in enumerate(y):
            ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        # 添加统计框
        if stats:
            text_lines = [
                f'Total Particles: {stats["total_particles"]}',
                f'Mean Diameter: {stats["mean_diameter"]:.2f} μm',
                f'Median (D50): {stats["d50"]:.2f} μm',
                f'Total Images: {stats["total_images"]}'
            ]
            if self.concentration != 0.0:
                text_lines.append(f'Concentration: {self.concentration:.4f} kg/m³')

            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.95, 0.95, '\n'.join(text_lines),
                    transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_xlabel("Particle Diameter Range (um)")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Particle Size Distribution")
        ax.grid(True)

        self.figure_diameter = fig
        return fig
    def plot_classification_results(self):
        if not self.particle_identification:
            return None

        class_info  = self.particle_identification
        class_names = list(class_info.keys())
        percents    = [info[0] for info in class_info.values()]
        counts      = [info[1] for info in class_info.values()]

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        x = np.arange(len(class_names))
        y = percents

        ax.plot(x, y, marker='s', linestyle='-', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")

        for i, (pct, cnt) in enumerate(zip(percents, counts)):
            ax.text(i, pct + 0.5, f'{cnt}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8)

        title = "Particle Classification"
        if self.method == 'polar':
            title += " (Amplitude Images)"

        ax.set_title(title)
        ax.set_xlabel("Particle Type")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, max(percents) * 1.2 if percents else 100)
        ax.grid(True)

        self.figure_classification = fig
        return fig

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