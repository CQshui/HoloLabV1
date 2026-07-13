from config import HoloConfig
from hologram import Hologram
from utils.open_image import OpenImage
from utils.preprocessing import PreProcessing
from utils.spectrum import Spectrum
from utils.reconstruction import Reconstruction, Reconstruction_DongJY
from utils.focusing import Focusing
from utils.segmentation import Segmentation
from utils.identification import Identification
from utils.phase import Phase
from utils.data_summary import DataSummary

from utils.polarization import Polarization
from utils.mock_data import MockData

'''MainWindow'''
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QMenu,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QCheckBox, QGroupBox, QMessageBox, QScrollArea,
    QSizePolicy, QTabWidget, QTextEdit, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSlider,
    QFileDialog, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QAction
from PyQt6.QtCore import Qt, QObject, QEvent, QTimer, QPoint, QThread, pyqtSignal, pyqtSlot

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
    'legend.fontsize': 10
})

import json
import sys
import os
import glob
import time
import numpy as np
from common.constants import unit_cm, unit_mm, unit_um, unit_nm
from PIL import Image
from typing import Union

class Holo_Processor(QObject):
    image_ready         = pyqtSignal(str, object, str)  # (步骤名称, 图像, 提示)
    save_load_operation = pyqtSignal(str, str)          # (步骤名称, 信息)

    def __init__(self, config: HoloConfig):
        super().__init__()

        self.config     = config
        self.hologram   = Hologram(self.config)
        self.images     = {}

        self._abort_requested = False   # 中途停止

    @pyqtSlot(str)
    def run(self, action_key: str):

        if   action_key == "open_image":
            work_open_image = OpenImage(hologram=self.hologram, config=self.config, mode = 'Single Image')
            work_open_image.run()
            self._show_hologram()

            # work_open_image = OpenImage(hologram=self.hologram ,config=self.config)
            # hologram_raw, hologram, _image_loaded, image_height, image_width, holo_type, status_msg = work_open_image.run()
            # work_open_image.run()

            # if hologram_raw is None:
            #     img_key  = "Origin"
            #     img_dict = {img_key: self.hologram.hologram_raw}
            #     self.image_ready.emit(img_key, img_dict, status_msg)
            #
            # else:
            #     self.hologram.hologram_raw  = hologram_raw
            #     self.hologram.hologram      = hologram
            #     self.hologram._image_loaded = _image_loaded
            #
            #     self.config.file_info['holo_type']    = holo_type
            #     self.config.image_info['pixel_num_x'] = image_height
            #     self.config.image_info['pixel_num_y'] = image_width
            #
            #     img_key = "Origin"
            #     img_dict = {img_key: self.hologram.hologram_raw}
            #
            #     self.images["Origin"] = img_dict
            #     self.image_ready.emit(img_key, img_dict, status_msg)

        elif action_key == "preprocessing":
            self.image_ready.emit("PreProcessing", None, f"PreProcessing method: {self.config.pre_process['method']}")

            worker_preprocessing = PreProcessing(hologram=self.hologram ,config=self.config)
            image, status_msg = worker_preprocessing.run()

            self.hologram.hologram = image

            img_key  = "PreProcessing"
            img      = self.hologram.hologram
            img_dict = {img_key: img}

            self.images[img_key] = img_dict
            self.image_ready.emit(img_key, img_dict, f"{img_key} Finished. {status_msg}")

        elif action_key == "polarization":
            self.image_ready.emit("Polarization", None, f"Split, Calculation")
            worker_polarization = Polarization(hologram=self.hologram, config=self.config)
            worker_polarization.run()
            self._show_polarization()
            self._show_hologram()

        elif action_key == "spectrum":
            self.image_ready.emit("Spectrum", None, f"Spectrum method: {self.config.spectrum['method']}")

            worker_spectrum = Spectrum(hologram=self.hologram, config=self.config)
            worker_spectrum.run()
            self._show_spectrum()

        elif action_key == "reconstruction":
            self.image_ready.emit("Reconstruction", {}, f"Reconstruction method: {self.config.reconstruction['method']}, On Going, Please wait ... ...")

            worker_reconstruction = Reconstruction(hologram=self.hologram ,config=self.config)
            worker_reconstruction.run()
            self._show_reconstruction()

        elif action_key == "focusing":
            self.image_ready.emit("Focusing", {}, f"Focusing method: {self.config.focusing['method']}, On Going, Please wait ... ...")

            # worker_focusing = Focusing(hologram=self.hologram ,config=self.config)
            worker_focusing = Focusing(hologram=self.hologram ,config=self.config)
            worker_focusing.run()
            self._show_focusing()

        elif action_key == "segmentation":
            self.image_ready.emit("Segmentation", {}, f"Segmentation method: {self.config.segmentation['method']}, On Going, Please wait ... ...")

            worker_segmentation = Segmentation(hologram=self.hologram ,config=self.config)
            worker_segmentation.run()
            self._show_segmentation()

        elif action_key == "identification":
            self.image_ready.emit("Identification", {}, f"Identification method: {self.config.identification['method']}, On Going, Please wait ... ...")

            worker_identification = Identification(hologram=self.hologram ,config=self.config)
            worker_identification.run()
            self._show_identification()

        elif action_key == "phase":
            self.image_ready.emit("Phase", {}, f"Phase analysis method: {self.config.phase['method']}, On Going, Please wait ... ...")

            worker_phase = Phase(hologram=self.hologram ,config=self.config)
            worker_phase.run()
            self._show_phase()

        elif action_key == "data_summary":
            worker_data_summary = DataSummary(hologram=self.hologram ,config=self.config)
            worker_data_summary.run()
            self._show_data_summary()

            '''频谱调试'''
            # fig = Figure(figsize=(6, 4))  # 可设置图像大小
            # ax1 = fig.add_subplot(121)  # 添加子图
            # log_spectrum_raw = np.log(1e-5 + np.abs(self.hologram.spectrum_raw))
            # im1 = ax1.imshow(log_spectrum_raw, cmap='viridis', aspect='auto')
            # ax1.set_title("2D Image Display")  # 设置标题
            #
            # ax2 = fig.add_subplot(122)  # 添加子图
            # log_spectrum = np.log(1e-5 + np.abs(self.hologram.spectrum))
            # im2 = ax2.imshow(log_spectrum, cmap='viridis', aspect='auto')
            # ax2.set_title("2D Image Display")  # 设置标题
            #
            # data_dict = {
            #     'Spec' : fig
            # }

            '''重建调试'''
            # key_list = list(self.hologram.reconstruction.keys())
            # fig = Figure(figsize=(6, 4))  # 可设置图像大小
            # ax1 = fig.add_subplot(121)  # 添加子图
            # log_spectrum_raw = np.abs(self.hologram.reconstruction[key_list[0]])
            # im1 = ax1.imshow(log_spectrum_raw, cmap='viridis', aspect='auto')
            # ax1.set_title(key_list[0])  # 设置标题
            #
            # ax2 = fig.add_subplot(122)  # 添加子图
            # log_spectrum = np.abs(self.hologram.reconstruction[key_list[20]])
            # im2 = ax2.imshow(log_spectrum, cmap='viridis', aspect='auto')
            # ax2.set_title(key_list[10])  # 设置标题
            #
            # data_dict = {
            #     'Spec' : fig
            # }

        elif action_key == "debug":
            ''''''
            pass

        # 单张图像：一键处理所有操作
        elif action_key == "all_in_one":
            if self.config.operation_options['run_open_image']:
                self.run("open_image")
            if self.config.operation_options['run_preprocessing']:
                self.run("preprocessing")
            if self.config.operation_options['run_polarization']:
                self.run("polarization")
            if self.config.operation_options['run_spectrum']:
                self.run("spectrum")
            if self.config.operation_options['run_reconstruction']:
                self.run("reconstruction")
            if self.config.operation_options['run_focusing']:
                self.run("focusing")
            if self.config.operation_options['run_segmentation']:
                self.run("segmentation")
            if self.config.operation_options['run_identification']:
                self.run("identification")
            if self.config.operation_options['run_phase']:
                self.run("phase")
            if self.config.operation_options['run_data_summary']:
                self.run("data_summary")
        # 单张图像：直接提取当前 Hologram 类中的图像、数据显示到GUI
        elif action_key == "refresh_image":
            '''Hologram Raw'''
            if 1:
                img_key  = "Origin"
                img      = self.hologram.hologram_raw
                img_dict = {img_key: img}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, "Refresh: Raw Hologram.")

            '''Hologram'''
            if 1:
                img_key = "PreProcessing"
                img = self.hologram.hologram
                img_dict = {img_key: img}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''Spectrum'''
            if 1:
                img_key = "Spectrum"
                img_dict = {'Spectrum'      : self.hologram.spectrum_raw,
                            'Spectrum Side' : self.hologram.spectrum,}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''Reconstruction'''
            if 1:
                self.images["Reconstruction"] = self.hologram.reconstruction
                self.image_ready.emit("Reconstruction", self.hologram.reconstruction, "Refresh: Reconstruction.")

            '''Focusing'''
            if 1:
                img_key  = "Focusing"
                img      = self.hologram.focusing
                img_dict = {img_key: img}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''Segmentation'''
            if 1:
                img_key  = "Segmentation"
                img      = np.abs(self.hologram.segmentation)
                img_dict = {img_key: img}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''Identification'''
            if 1:
                img_key  = "Identification"
                img      = np.abs(self.hologram.identification)
                img_dict = {img_key: img}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''Phase'''
            if 1:
                img_key  = "Phase"
                img_dict = {'Origin'        : self.hologram.phase,
                            'Unwrapped'     : self.hologram.phase_unwrapped,
                            'Compensated'   : self.hologram.phase_compensated,
                            'Compensation'  : self.hologram.phase_compensation_mat,
                            'Corrected'     : self.hologram.phase_corrected}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''Polarization'''
            if 1:

                img_key = "Polarization"
                img_dict = {'Angle 0'   : self.hologram.hologram_p000 // 2,
                            'Angle 90'  : self.hologram.hologram_p045 // 4,
                            'Angle 135' : self.hologram.hologram_p090 // 5,
                            'Angle 180' : self.hologram.hologram_p135 // 8}

                self.images[img_key] = img_dict
                self.image_ready.emit(img_key, img_dict, f"Refresh: {img_key}.")

            '''data_summary'''
            if 1:
                ''''''
                pass
        # 多张图像
        elif action_key == "multi_images":
            self._abort_requested = False  # 每次运行前清除旧的中止请求

            if self.config.multi_processing['run_open_image']:
                images_path     = self.config.multi_processing['images_path']

                work_open_image = OpenImage(hologram=self.hologram, config=self.config, mode = 'Multi Images')
                work_open_image.run()

                images_urls     = self.config.multi_processing['images_urls']
                non_images_urls = self.config.multi_processing['non_images_urls']

                if not images_urls:
                    self.image_ready.emit("Multi Processing", np.array([len(images_urls)]), f"No images found in {images_path}.")
                    return
                else:
                    self.image_ready.emit("Multi Processing", np.array([len(images_urls)]), f"{len(images_urls)} images found in {images_path}")

            max_handle_num = self.config.multi_processing['max_handle_num']
            max_handle_num = max_handle_num if max_handle_num is not None else 999999999

            images_num     = len(images_urls)
            for i, image in enumerate(images_urls):

                self.config.file_info['image_path'] = os.path.dirname(image)
                self.config.file_info['image_name'] = os.path.basename(image)

                if self.config.multi_processing['run_open_image']:
                    work_open_image = OpenImage(hologram=self.hologram, config=self.config, mode='Single Image')
                    work_open_image.run()
                    self.image_ready.emit("Multi Processing", {}, f"  {i+1} / {images_num}, {image}")

                if self.config.multi_processing['run_preprocessing']:
                    worker_preprocessing = PreProcessing(hologram=self.hologram, config=self.config)
                    self.hologram.hologram, _ = worker_preprocessing.run()
                    self.image_ready.emit("Multi Processing", {}, 'Pre Set > ')

                if self.config.multi_processing['run_polarization']:
                    worker_polarization = Polarization(hologram=self.hologram, config=self.config)
                    worker_polarization.run()
                    self.image_ready.emit("Multi Processing", {}, 'Polar > ')

                if self.config.multi_processing['run_spectrum']:
                    worker_spectrum = Spectrum(hologram=self.hologram, config=self.config)
                    worker_spectrum.run()
                    self.image_ready.emit("Multi Processing", {}, 'Spectrum > ')

                if self.config.multi_processing['run_reconstruction']:
                    woker_reconstruction = Reconstruction(hologram=self.hologram, config=self.config)
                    woker_reconstruction.run()
                    self.image_ready.emit("Multi Processing", {}, 'Reconstruct > ')

                if self.config.multi_processing['run_focusing']:
                    worker_focusing = Focusing(hologram=self.hologram ,config=self.config)
                    worker_focusing.run()
                    self.image_ready.emit("Multi Processing", {}, 'Focusing > ')

                if self.config.multi_processing['run_segmentation']:
                    worker_segmentation = Segmentation(hologram=self.hologram ,config=self.config)
                    worker_segmentation.run()
                    self.image_ready.emit("Multi Processing", {}, 'Segment > ')

                if self.config.multi_processing['run_identification']:
                    worker_identification = Identification(hologram=self.hologram ,config=self.config)
                    worker_identification.run()
                    self.image_ready.emit("Multi Processing", {}, 'Identify > ')

                if self.config.multi_processing['run_phase']:
                    worker_phase = Phase(hologram=self.hologram ,config=self.config)
                    self.hologram.phase = worker_identification.run()
                    self.image_ready.emit("Multi Processing", {}, 'Phase > ')

                if self.config.multi_processing['run_data_summary']:
                    worker_data_summary = DataSummary(hologram=self.hologram ,config=self.config)
                    worker_data_summary.run()
                    self.image_ready.emit("Multi Processing", {}, 'Analysis > ')

                self.image_ready.emit("Multi Processing", [i + 1, images_num], 'Done.')   # 用list作为单幅图像处理完成的信号
                np.array([i + 1, images_num])
                if self._check_abort():
                    self.image_ready.emit("Multi Processing", np.array([images_num]), f"Multi Processing Aborted.")
                    return

                if i+1 == max_handle_num:
                    break

            self.image_ready.emit("Multi Processing", np.array([i + 1, images_num]), f"Finished. Total {i+1} / {images_num}.")
        # 保存/读取配置文件、数据操作
        elif action_key == "save_config":
            config_path = self.hologram.config.save_and_load['config_save_path']
            config_name = self.hologram.config.save_and_load['config_save_name']

            config_url = os.path.join(config_path, config_name)
            msg = self.hologram.config.save_config(config_url)

            self.save_load_operation.emit("Save Config", msg)

        elif action_key == "load_config":
            config_path = self.hologram.config.save_and_load['config_load_path']
            config_name = self.hologram.config.save_and_load['config_load_name']

            config_url = os.path.join(config_path, config_name)
            msg = self.hologram.config.load_config(config_url)

            self.save_load_operation.emit("Load Config", msg)

        elif action_key == "save_data":
            msg = self.hologram.save_data()
            self.save_load_operation.emit("Save Data", msg)

        elif action_key == "load_data":
            msg = self.hologram.load_data()
            self.save_load_operation.emit("Load Data", msg)

        else:
            self.image_ready.emit(img_key, None, "Unknown action: {action_key}.")

    def _show_hologram(self):
        img_key = "Origin"
        img_dict = {
            'Raw Image': self.hologram.hologram_raw,
            'Processing': self.hologram.hologram}

        self.images["Origin"] = img_dict
        self.image_ready.emit(img_key, img_dict, self.hologram.status_msg)
    def _show_polarization(self):
        img_key = "Polarization"
        img_dict = {'Angle 0': self.hologram.hologram_p000,
                    'Angle 45': self.hologram.hologram_p045,
                    'Angle 90': self.hologram.hologram_p090,
                    'Angle 135': self.hologram.hologram_p135,
                    'S0': self.hologram.polar_S0,
                    'S1': self.hologram.polar_S1,
                    'S2': self.hologram.polar_S2,
                    'Amplitude': self.hologram.polar_amp}

        self.images[img_key] = img_dict
        self.image_ready.emit(img_key, img_dict, self.hologram.status_msg)
    def _show_spectrum(self):
        img_key = "Spectrum"
        img_dict = {
            'Initial (Log10)': np.log10(0.0000001 + np.abs(self.hologram.spectrum_raw)),
            'Process (Log10)': np.log10(0.0000001 + np.abs(self.hologram.spectrum))
        }

        self.images[img_key] = img_dict
        self.image_ready.emit(img_key, img_dict, self.hologram.status_msg)
    def _show_reconstruction(self):
        '归一化'
        reconstruction = self.hologram.reconstruction
        for _key, _value in reconstruction.items():
            abs_v = np.abs(_value.copy())
            min_val = np.min(abs_v)
            max_val = np.max(abs_v)
            reconstruction[_key] = (abs_v - min_val) / (max_val - min_val)

        self.images["Reconstruction"] = reconstruction
        self.image_ready.emit("Reconstruction", reconstruction, self.hologram.status_msg)  # 显示等待动画等
    def _show_focusing(self):
        if self.hologram.focusing_each == {}:
            img_key = "Focusing"
            img = self.hologram.focusing
            img_dict = {img_key: img}
        else:
            '''展示总体、个体'''
            img_dict = self.hologram.focusing_each

            img_key = "Focusing"
            img = np.abs(self.hologram.focusing)

            img_dict.pop(img_key, None)  # 如果 key 已存在，先删除它（避免重复）
            img_dict = {img_key: img, **img_dict}  # 重建字典，确保新键在最前面

        self.images["Focusing"] = img_dict
        self.image_ready.emit(img_key, img_dict, self.hologram.status_msg)  # 显示等待动画等
    def _show_segmentation(self):
        if self.hologram.segmentation_each == {}:
            img_key = "Segmentation"
            img = self.hologram.segmentation
            img_dict = {img_key: img}
        else:
            '''展示总体、个体'''
            img_dict = self.hologram.segmentation_each

            img_key = "Segmentation"
            img = np.abs(self.hologram.segmentation)

            img_dict.pop(img_key, None)  # 如果 key 已存在，先删除它（避免重复）
            img_dict = {img_key: img, **img_dict}  # 重建字典，确保新键在最前面

        self.image_ready.emit(img_key, img_dict, self.hologram.status_msg)
    def _show_identification(self):
        identification = self.hologram.identification_each

        self.images["Identification"] = identification
        self.image_ready.emit("Identification", identification, self.hologram.status_msg)
    def _show_phase(self):
        img_key = "Phase"
        img_dict = {'Origin':       self.hologram.phase,
                    'Unwrapped':    self.hologram.phase_unwrapped,
                    'Compensated':  self.hologram.phase_compensated,
                    'Compensation': self.hologram.phase_compensation_mat,
                    'Corrected':    self.hologram.phase_corrected}

        self.images[img_key] = img_dict
        self.image_ready.emit(img_key, img_dict, "Phase Analysis Finished.")
    def _show_data_summary(self):
        data_dict = {}
        data_dict['Diameter'] = self.hologram.figure_diameter
        data_dict['Classification'] = self.hologram.figure_classification

        self.image_ready.emit('Analysis', data_dict, "Data Analysis Finished.")

    def stop(self):
        """外部调用，用于请求中止当前处理"""
        self._abort_requested = True

    def _check_abort(self):
        """供 run 内部周期性检查中止请求"""
        return self._abort_requested

class Holo_Controller(QObject):
    run_signal = pyqtSignal(str)

    def __init__(self, gui, processor: Holo_Processor):
        super().__init__()
        self.gui = gui
        self.processor = processor

        # connect signal to processor slot
        self.run_signal.connect(self.processor.run)
        self.connect_signals()

        self.processor.image_ready.connect(self.handle_image_ready)
        self.processor.save_load_operation.connect(self.handle_save_load_operation)

        self.gui.btn_multi_images_stop.clicked.connect(self.stop)

    def connect_signals(self):
        ''''''
        '''参数变化'''
        self.gui.param_changed.connect(self.update_config)

        '''按钮操作'''
        self.gui.btn_refresh_image.clicked.connect(lambda: self.run_signal.emit("refresh_image"))

        self.gui.btn_open_image.clicked.connect(lambda: self.run_signal.emit("open_image"))
        self.gui.btn_preprocessing.clicked.connect(lambda: self.run_signal.emit("preprocessing"))
        self.gui.btn_spectrum.clicked.connect(lambda: self.run_signal.emit("spectrum"))
        self.gui.btn_reconstruction.clicked.connect(lambda: self.run_signal.emit("reconstruction"))
        self.gui.btn_focusing.clicked.connect(lambda: self.run_signal.emit("focusing"))
        self.gui.btn_segmentation.clicked.connect(lambda: self.run_signal.emit("segmentation"))
        self.gui.btn_identification.clicked.connect(lambda: self.run_signal.emit("identification"))
        self.gui.btn_phase.clicked.connect(lambda: self.run_signal.emit("phase"))
        self.gui.btn_data_summary.clicked.connect(lambda: self.run_signal.emit("data_summary"))
        self.gui.btn_polarization.clicked.connect(lambda: self.run_signal.emit("polarization"))
        self.gui.btn_all_in_one.clicked.connect(lambda: self.run_signal.emit("all_in_one"))

        self.gui.btn_save_config.clicked.connect(lambda: self.run_signal.emit("save_config"))
        self.gui.btn_load_config.clicked.connect(lambda: self.run_signal.emit("load_config"))
        self.gui.btn_save_data.clicked.connect(lambda: self.run_signal.emit("save_data"))
        self.gui.btn_load_data.clicked.connect(lambda: self.run_signal.emit("load_data"))

        self.gui.btn_multi_images_start.clicked.connect(lambda: self.run_signal.emit("multi_images"))
        self.gui.btn_multi_images_stop.clicked.connect(lambda: self.stop)

    def update_config(self, key_path, value):
        parts = key_path.split(".")
        try:
            if len(parts) == 1:
                # 顶层属性
                setattr(self.processor.config, parts[0], value)

            elif len(parts) == 2:
                # 一级字典，如 file_info.save_path
                dict_obj = getattr(self.processor.config, parts[0])
                if isinstance(dict_obj, dict):
                    dict_obj[parts[1]] = value
                else:
                    raise TypeError(f"{parts[0]} 不是字典类型")

            elif len(parts) == 3:
                # 二级嵌套字典，如 file_info.ROI_rectangle.center_x
                dict_obj = getattr(self.processor.config, parts[0])
                if isinstance(dict_obj, dict):
                    nested_dict = dict_obj.get(parts[1])
                    if isinstance(nested_dict, dict):
                        nested_dict[parts[2]] = value
                    else:
                        raise TypeError(f"{parts[0]}.{parts[1]} 不是字典类型")
                else:
                    raise TypeError(f"{parts[0]} 不是字典类型")

            else:
                # 3 层以上嵌套字典，暂不支持
                raise NotImplementedError("暂不支持 3 层以上嵌套")

            # print(f"[Controller] Parameter updated: {key_path} = {value}")

        except Exception as e:
            print(f"[Controller] Failed to update parameter: {key_path} -> {e}")

    def handle_image_ready(self, operation_name: str, image: np.ndarray, status: str):
        self.gui.update_image(operation_name, image, status)

    def handle_save_load_operation(self, operation_name: str, message: str):
        self.gui.log_on_save_load(operation_name, message)

    def stop(self):
        self.processor.stop()

class Collapsible_Section(QWidget):
    def __init__(self, title, title_color="black", parent=None, expanded=False, show_button=True):
        """
        可折叠的区域组件

        Args:
            title (str): 标题文本
            title_color (str): 标题颜色，默认黑色
            parent: 父组件
            expanded (bool): 初始状态是否展开，默认False（折叠）
            show_button (bool): 是否显示标题按钮，默认True（显示）
        """
        super().__init__(parent)

        self.expanded = expanded
        self.show_button = show_button

        # 创建切换按钮
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(self.expanded)
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {title_color};
                color       : white;
                padding     : 5px;
                padding-left: 10px;  /* 加这个就是缩进 */
                text-align  : left;
                border      : none;
                border-radius: 3px;  /* 加上圆角 */
                height: 23px;
                font-size   : 13px;
                font-weight : bold;
                /* font-family: "Arial"; */
            }}
            QPushButton:checked {{
                background-color: {title_color};  /* 不变色 */
            }}
            QPushButton:hover {{
                background-color: {title_color};  /* 不变色 */
            }}
        """)

        # 设置按钮的可见性和高度
        if not self.show_button:
            self.toggle_button.setVisible(False)
            self.toggle_button.setMaximumHeight(0)
            self.toggle_button.setMinimumHeight(0)

        # 创建内容区域
        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # 根据初始状态设置内容区域的高度
        if self.expanded:
            self.content_area.setMaximumHeight(16777215)  # Qt的最大高度值
        else:
            self.content_area.setMaximumHeight(0)

        # 连接信号
        self.toggle_button.toggled.connect(self.toggle_content)

        # 设置主布局 - 始终添加所有组件，避免动态布局变化
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.toggle_button)  # 始终添加，通过可见性控制
        self.main_layout.addWidget(self.content_area)

        # 设置内容布局
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 0, 0) # # 左、上、右、下
        self.content_area.setLayout(self.content_layout)

    def toggle_content(self, checked):
        """切换内容区域的显示/隐藏"""
        self.expanded = checked
        if checked:
            self.content_area.setMaximumHeight(16777215)
        else:
            self.content_area.setMaximumHeight(0)

    def add_widget(self, widget):
        """添加组件到内容区域"""
        self.content_layout.addWidget(widget)

    def set_expanded(self, expanded):
        """程序化设置展开/折叠状态"""
        self.expanded = expanded
        if self.show_button:
            self.toggle_button.setChecked(expanded)
        else:
            # 如果按钮隐藏，直接更新内容区域
            if expanded:
                self.content_area.setMaximumHeight(16777215)
            else:
                self.content_area.setMaximumHeight(0)

    def is_expanded(self):
        """返回当前是否展开"""
        return self.expanded

    def set_button_visible(self, visible):
        """设置按钮的可见性"""
        self.show_button = visible
        self.toggle_button.setVisible(visible)

        # 通过设置高度来控制按钮占用空间，避免动态布局变化
        if not visible:
            self.toggle_button.setMaximumHeight(0)
            self.toggle_button.setMinimumHeight(0)
        else:
            self.toggle_button.setMaximumHeight(23)  # 恢复原始高度
            self.toggle_button.setMinimumHeight(23)

    def set_title(self, title):
        """设置标题文本"""
        self.toggle_button.setText(title)

    def get_title(self):
        """获取标题文本"""
        return self.toggle_button.text()

class Image_Viewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom       = 0
        self._empty      = True
        self.image_dict  = {}            # 用字典存储图像
        self.keys        = []            # 字典 keys 的顺序列表
        self.image_index = 0

        self._need_fit_on_show = False  #

        'Main layout'
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        'Image viewer area'
        self.viewer = QGraphicsView(self)
        self.viewer.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.viewer.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.viewer.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewer.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.viewer.setBackgroundBrush(QBrush(QColor(220, 220, 220)))
        self.viewer.setMouseTracking(True)
        self.layout.addWidget(self.viewer)

        self.scene = QGraphicsScene(self)
        self.viewer.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        'Slider bar (top)'
        # 图像标签
        self.slider_widget = QWidget(self)
        self.slider_layout = QHBoxLayout(self.slider_widget)
        self.slider_layout.setContentsMargins(5, 5, 5, 0)
        self.index_label = QLabel("", self)    # 显示当前key，不是数字索引
        self.index_label.setStyleSheet(
            "font-family: Arial; font-size: 14px; font-weight: bold"
        )
        self.index_label.setFixedWidth(120)

        # 图像滑块
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 5px;
                background: #ccc;
                border-radius: 5px;
                margin: 0px;
            }
            QSlider::handle:horizontal {
                background  : #000000;
                border      : 0px solid #555;
                width       : 10px;
                height      : 10px;
                margin      : -10px 0;
                border-radius: 5px;     /* 圆角半径 = 宽/2，形成圆形 */
            }
        """)

        self.slider.setMaximumWidth(300)        # 缩短进度条长度
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.installEventFilter(self)    # 用于滚轮事件

        self.slider_layout.addStretch()
        self.slider_layout.addWidget(self.index_label)
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addStretch()
        self.slider_widget.hide()
        self.layout.addWidget(self.slider_widget)

        'Pixel info label'
        self.pixel_label = QLabel(self.viewer)
        self.pixel_label.installEventFilter(self)
        self.pixel_label.setStyleSheet(
            "background-color: rgba(255,255,255,200); "
            "padding: 2px; border: 0px solid gray; "
            "font-family: Consolas; font-size: 11px;"
        )
        self.pixel_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_label.setFixedSize(180, 20)
        self.pixel_label.hide()

        self.image = None
        self.viewer.viewport().installEventFilter(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._move_pixel_label_to_corner()

    def _move_pixel_label_to_corner(self):
        """统一右下角定位像素标签"""
        self.pixel_label.move(
            self.viewer.viewport().width() - self.pixel_label.width() - 10,
            self.viewer.viewport().height() - self.pixel_label.height() - 10
        )

    def eventFilter(self, obj, event):
        # 滚动切图：鼠标滚轮在 slider 上滚动
        if obj == self.slider and event.type() == QEvent.Type.Wheel:
            if event.angleDelta().y() > 0:
                self.show_previous()
            else:
                self.show_next()
            return True

        # 图像区域滚动：缩放 或 Shift+滚动换图
        elif obj == self.viewer.viewport() and event.type() == QEvent.Type.Wheel:
            self._handle_wheel(event)
            return True

        # 鼠标移动：更新像素信息
        elif obj == self.viewer.viewport() and event.type() == QEvent.Type.MouseMove:
            self._handle_mouse_move(event)

        # 双击像素信息框：隐藏
        elif getattr(self, "pixel_label", None) is not None \
                and obj == self.pixel_label \
                and event.type() == QEvent.Type.MouseButtonDblClick:
            self.pixel_label.hide()
            return True

        return super().eventFilter(obj, event)

    def _handle_mouse_move(self, event):
        if self._empty or self.image is None:
            return

        scene_pos = self.viewer.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())

        '鼠标放置时，显示的是转换图像的像素值'
        # if 0 <= x < self.image.width() and 0 <= y < self.image.height():
        #     color = self.image.pixelColor(x, y)
        #     if color.isValid():
        #         if not self.pixel_label.isVisible():
        #             self.pixel_label.show()
        #
        #         if self.image.format() in [QImage.Format.Format_Grayscale8, QImage.Format.Format_Grayscale16]:
        #             text = f"({x}, {y}) | {color.red()}"
        #         else:
        #             text = f"({x}, {y}) | {color.red()}, {color.green()}, {color.blue()}"
        #         self.pixel_label.setText(text)
        #
        #         # 添加位置同步检测（防止错位）
        #         expected_x = self.viewer.viewport().width() - self.pixel_label.width() - 10
        #         expected_y = self.viewer.viewport().height() - self.pixel_label.height() - 10
        #         if self.pixel_label.pos() != QPoint(expected_x, expected_y):
        #             self._move_pixel_label_to_corner()

        '鼠标放置时，显示原始图像数据的像素值'
        if hasattr(self, "_raw_image") and self._raw_image is not None:
            img = self._raw_image
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                value = img[y, x]  # 注意 y 是行，x 是列
                if isinstance(value, np.generic):
                    value = value.item()  # 转成 Python 原生类型

                text = f"({x}, {y}) | {value:.4f}"
                self.pixel_label.setText(text)

                if not self.pixel_label.isVisible():
                    self.pixel_label.show()

                # 保证像素标签定位到右下角
                expected_x = self.viewer.viewport().width() - self.pixel_label.width() - 10
                expected_y = self.viewer.viewport().height() - self.pixel_label.height() - 10
                if self.pixel_label.pos() != QPoint(expected_x, expected_y):
                    self._move_pixel_label_to_corner()

        else:
            # fallback：使用 QImage 提取像素灰度或 RGB 值（用于非 float 类型）
            if 0 <= x < self.image.width() and 0 <= y < self.image.height():
                color = self.image.pixelColor(x, y)
                if color.isValid():
                    if self.image.format() in [QImage.Format.Format_Grayscale8, QImage.Format.Format_Grayscale16]:
                        text = f"({x}, {y}) | {color.red()}"
                    else:
                        text = f"({x}, {y}) | {color.red()}, {color.green()}, {color.blue()}"
                    self.pixel_label.setText(text)

                    if not self.pixel_label.isVisible():
                        self.pixel_label.show()

                    expected_x = self.viewer.viewport().width() - self.pixel_label.width() - 10
                    expected_y = self.viewer.viewport().height() - self.pixel_label.height() - 10
                    if self.pixel_label.pos() != QPoint(expected_x, expected_y):
                        self._move_pixel_label_to_corner()

    # 滚轮缩放、 Control+滚轮浏览
    def _handle_wheel_(self, event):
        # todo: 缩放百分比提示、复位缩放按钮

        if self._empty:
            return

        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.show_previous()
            else:
                self.show_next()
            return

        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8

        if angle > 0:
            self._zoom += 1
            self.viewer.scale(factor, factor)
        else:
            if self._zoom > -5:  # 给一个最小限制，防止缩太小
                self._zoom -= 1
                self.viewer.scale(factor, factor)
    # 滚轮浏览、 Control+滚轮缩放
    def _handle_wheel(self, event):
        # todo: 缩放百分比提示、复位缩放按钮

        if self._empty:
            return

        # ✅ 改为：不按Control时换图，按Control时缩放
        if event.modifiers() != Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.show_previous()
            else:
                self.show_next()
            return

        # 缩放行为（Control+滚轮）
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8

        if angle > 0:
            self._zoom += 1
            self.viewer.scale(factor, factor)
        else:
            if self._zoom > -5:  # 最小限制，防止缩太小
                self._zoom -= 1
                self.viewer.scale(factor, factor)

    def contextMenuEvent(self, event):
        if self._empty or self.image is None:
            return

        menu = QMenu(self)
        save_action = QAction("Save Image", self)
        save_action.triggered.connect(self._save_current_image)
        menu.addAction(save_action)
        menu.exec(event.globalPos())

    def _save_current_image(self):
        # 弹出文件夹选择框
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not folder:
            return

        key = self.keys[self.image_index]
        filename = f"{str(key)}.png"
        path = os.path.join(folder, filename)

        if self.image.save(path):
            QMessageBox.information(self, "Saved", f"Image save to:\n{path}")
        else:
            QMessageBox.warning(self, "Failed", "Fail to save image.")

    def set_image(self, image_dict: dict):
        if not image_dict:
            return

        self.image_dict  = image_dict
        self.keys        = list(image_dict.keys())
        self.image_index = 0
        self._empty = False

        if len(self.keys) > 1:
            self.slider_widget.show()
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.keys) - 1)
            self.slider.setValue(0)
            self.index_label.setText(str(self.keys[0]))
        else:
            self.slider_widget.hide()
            self.index_label.setText(str(self.keys[0]))

        self._show_image_at_index(0)

        # 如果当前可见，立即执行 fit
        if self.isVisible():
            self._fit_in_view()
            self._need_fit_on_show = False
        else:
            self._need_fit_on_show = True

    def _show_image_at_index(self, index: int):
        if 0 <= index < len(self.keys):
            key = self.keys[index]
            img = self.image_dict[key]

            self._raw_image = img  # 保存原始数据

            # 确保图像为 uint8
            if img.dtype != np.uint8:
                # 防止全 0 或常量图导致除以 0
                min_val, max_val = np.min(img), np.max(img)
                if max_val - min_val == 0:
                    norm_img = np.zeros_like(img, dtype=np.uint8)
                else:
                    norm_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                norm_img = img

            h, w = norm_img.shape

            # 这里假设输入是单通道灰度图 numpy.ndarray，格式如原来
            qimage = QImage(norm_img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            # qimage = QImage(norm_img.data, w, h, w, QImage.Format.Format_Grayscale8)
            self.image = qimage
            pixmap = QPixmap.fromImage(qimage)
            self.pixmap_item.setPixmap(pixmap)

            self._fit_in_view()

            self._zoom = 0
            self.index_label.setText(str(key))  # 显示字典key

            font_metrics = self.index_label.fontMetrics()
            text_width = font_metrics.horizontalAdvance(str(key)) + 10
            self.index_label.setFixedWidth(max(50, text_width))  # 保持至少120像素宽度

            self._move_pixel_label_to_corner()  # 保证像素标签位置始终正确

    def _fit_in_view(self):
        rect = self.pixmap_item.boundingRect()
        if not rect.isNull():
            self.viewer.setSceneRect(rect)
            self.viewer.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def show_next(self):
        if self.image_index + 1 < len(self.keys):
            self.image_index += 1
            self.slider.setValue(self.image_index)

    def show_previous(self):
        if self.image_index - 1 >= 0:
            self.image_index -= 1
            self.slider.setValue(self.image_index)

    def _on_slider_changed(self, value):
        self.image_index = value
        self._show_image_at_index(value)

    def showEvent(self, event):
        super().showEvent(event)
        if self._need_fit_on_show:
            self._fit_in_view()
            self._move_pixel_label_to_corner()
            self._need_fit_on_show = False

class Data_Viewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._empty = True

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)      # 左、上、右、下

        # matplotlib 工具栏和画布
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 底部按钮区域，用于切换图像
        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(20, 0, 20, 0)   # 左、上、右、下
        self.button_layout.setSpacing(3)
        self.button_widget.setMaximumHeight(50)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.button_widget)

        self.setLayout(self.layout)

        # 数据存储
        self.data_dict = {}
        self.keys = []
        self.data_index = 0

        # 按钮列表
        self.buttons = []

    def set_image(self, data_dict: dict):
        """
        显示 matplotlib 图像，格式要求为 {"title" 或其他 key: Figure 实例}
        """
        if not data_dict:
            return

        if not isinstance(data_dict, dict):
            raise ValueError("set_image 参数必须是 dict")

        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.data_index = 0
        self._empty = False

        # 清空旧按钮
        for btn in self.buttons:
            btn.deleteLater()
        self.buttons.clear()

        # 创建新按钮
        for i, key in enumerate(self.keys):
            btn = QPushButton(str(key), self.button_widget)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._show_figure_at_index(idx))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E0E0E0;
                    color: #333;
                    border: none;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size: 13px;
                    font-family: Arial;
                }
                QPushButton:hover {
                    background-color: #CCCCCC;
                }
                QPushButton:pressed {
                    background-color: #A0A0A0;
                }
                QPushButton:disabled {
                    background-color: #909090;
                    color: #F0F0F0;
                }
            """)
            btn.setMinimumHeight(35)
            btn.setMaximumWidth(250)

            self.button_layout.addWidget(btn)
            self.buttons.append(btn)

        # 默认选中第一个按钮
        if self.buttons:
            self.buttons[0].setChecked(True)

        self._show_figure_at_index(0)
    def _show_figure_at_index(self, index: int):
        if 0 <= index < len(self.keys):
            key = self.keys[index]
            fig = self.data_dict[key]
            if not isinstance(fig, Figure):
                raise ValueError(f"key={key} corresponding data is not a 'Figure' object.")

            self.figure.clear()
            for src_ax in fig.axes:
                dst_ax = self.figure.add_subplot(src_ax.get_subplotspec())
                self._copy_axes_content(src_ax, dst_ax)
            self.canvas.draw()

            # 更新按钮状态
            self.data_index = index
            for i, btn in enumerate(self.buttons):
                btn.setChecked(i == index)
    def _copy_axes_content(self, src_ax, dst_ax):
        for line in src_ax.get_lines():
            dst_ax.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        marker=line.get_marker(),
                        linewidth=line.get_linewidth())
        for image in src_ax.get_images():
            dst_ax.imshow(image.get_array(),
                          cmap=image.get_cmap(),
                          extent=image.get_extent(),
                          origin=image.origin,
                          interpolation=image.get_interpolation())
        for patch in src_ax.patches:
            dst_ax.add_patch(patch)

        dst_ax.set_title(src_ax.get_title())
        dst_ax.set_xlabel(src_ax.get_xlabel())
        dst_ax.set_ylabel(src_ax.get_ylabel())
        dst_ax.set_xlim(src_ax.get_xlim())
        dst_ax.set_ylim(src_ax.get_ylim())

        dst_ax.grid(any(line.get_visible() for line in src_ax.get_xgridlines() + src_ax.get_ygridlines()))

    def set_image_(self, data_dict: dict):
        if not data_dict:
            return

        if not isinstance(data_dict, dict):
            raise ValueError("set_image 参数必须是 dict")

        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.data_index = 0
        self._empty = False

        # 清空旧按钮
        for btn in self.buttons:
            btn.deleteLater()
        self.buttons.clear()

        for i, key in enumerate(self.keys):
            btn = QPushButton(str(key), self.button_widget)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._show_figure_at_index(idx))
            btn.setMinimumHeight(35)
            btn.setMaximumWidth(250)
            self.button_layout.addWidget(btn)
            self.buttons.append(btn)

        if self.buttons:
            self.buttons[0].setChecked(True)

        self._show_figure_at_index(0)
    def _show_figure_at_index_(self, index: int):
        if 0 <= index < len(self.keys):
            key = self.keys[index]
            fig = self.data_dict[key]
            if not isinstance(fig, Figure):
                raise ValueError(f"key={key} 对应数据不是 Figure 实例")

            self.figure.clear()

            # 复制 axes 内容到当前唯一 figure
            for src_ax in fig.axes:
                dst_ax = self.figure.add_subplot(src_ax.get_subplotspec())
                self._copy_axes_content_safe(src_ax, dst_ax)

            self.canvas.draw()

            self.data_index = index
            for i, btn in enumerate(self.buttons):
                btn.setChecked(i == index)
    def _copy_axes_content_safe(self, src_ax, dst_ax):
        # 拷贝线条
        for line in src_ax.get_lines():
            dst_ax.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        marker=line.get_marker(),
                        linewidth=line.get_linewidth())

        # 拷贝图像（如imshow）
        for image in src_ax.get_images():
            dst_ax.imshow(image.get_array(),
                          cmap=image.get_cmap(),
                          extent=image.get_extent(),
                          origin=image.origin,
                          interpolation=image.get_interpolation())

        # 不拷贝 bar patch（patch 不能跨 figure），可选用原始数据重绘方式

        # 拷贝标题、坐标轴信息
        dst_ax.set_title(src_ax.get_title())
        dst_ax.set_xlabel(src_ax.get_xlabel())
        dst_ax.set_ylabel(src_ax.get_ylabel())
        dst_ax.set_xlim(src_ax.get_xlim())
        dst_ax.set_ylim(src_ax.get_ylim())
        dst_ax.set_xticks(src_ax.get_xticks())
        dst_ax.set_xticklabels([label.get_text() for label in src_ax.get_xticklabels()],
                               rotation=45, ha="right")  # 可调整

        dst_ax.grid(any(line.get_visible() for line in src_ax.get_xgridlines() + src_ax.get_ygridlines()))

    def clear(self):
        self.figure.clear()
        self.canvas.draw()
        # 清空按钮
        for btn in self.buttons:
            btn.deleteLater()
        self.buttons.clear()
        self._empty = True

    def get_toolbar(self):
        """返回 matplotlib 工具栏实例"""
        return self.toolbar

class MainWindow(QMainWindow):
    param_changed = pyqtSignal(str, object)

    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("HoloLab")
        self.input_fields = {}

        self.config = config
        # self.load_config()

        self.init_ui()

        self.param_changed.connect(self.log_on_param_changed)
        self._operation_seconds = 0     # 用于计时器累加

    def init_ui(self):
        ''''''
        '''菜单栏'''
        self._init_menu_bar()

        '''状态栏'''
        self.statusBar().showMessage("HoloLab")

        central      = QWidget()
        main_layout  = QHBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        '''左中右布局部分'''
        left_panel   = self._init_left_panel()
        middle_panel = self._init_middle_panel()
        right_panel  = self._init_right_panel()

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(middle_panel, 5)
        main_layout.addWidget(right_panel, 1)

    def _init_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        'Mode 菜单'
        mode_menu = menubar.addMenu("Mode")
        single_action = QAction("Single Mode", self)
        single_action.triggered.connect(lambda: self.mode_tab_widget.setCurrentIndex(0))
        single_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(0, True))
        single_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(1, False))
        single_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(2, False))

        mode_menu.addAction(single_action)

        multi_action = QAction("Multi Mode", self)
        multi_action.triggered.connect(lambda: self.mode_tab_widget.setCurrentIndex(1))
        multi_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(0, False))
        multi_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(1, True))
        multi_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(2, False))

        mode_menu.addAction(multi_action)

        camera_action = QAction("Camera Mode", self)
        camera_action.triggered.connect(lambda: self.mode_tab_widget.setCurrentIndex(2))
        camera_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(0, False))
        camera_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(1, False))
        camera_action.triggered.connect(lambda: self.mode_tab_widget.setTabVisible(2, True))

        mode_menu.addAction(camera_action)

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Hide/Show Settings")
        view_menu.addAction("Hide/Show Operation")

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Doc")
        help_menu.addAction("About")
    def _init_left_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(345)

        param_container = QWidget()
        param_layout    = QVBoxLayout(param_container)
        param_layout.setContentsMargins(3, 0, 3, 0)     # 左、上、右、下

        '''参数设置'''
        param_layout.addWidget(self.create_file_info_group())
        param_layout.addWidget(self.create_image_info_group())
        param_layout.addWidget(self.create_preprocess_group())
        param_layout.addWidget(self.create_spectrum_group())
        param_layout.addWidget(self.create_reconstruction_group())
        param_layout.addWidget(self.create_focusing_group())
        param_layout.addWidget(self.create_segmentation_group())
        param_layout.addWidget(self.create_identification_group())
        param_layout.addWidget(self.create_phase_group())
        param_layout.addWidget(self.create_polarization_group())
        param_layout.addWidget(self.create_data_summary_group())
        param_layout.addWidget(self.create_save_and_load_group())
        param_layout.addStretch()

        '''保存/导出 按钮'''
        button_grid = QGridLayout()
        self.btn_save_config    = QPushButton("Save Config")
        self.btn_load_config    = QPushButton("Load Config")
        self.btn_save_data      = QPushButton("Save Data")
        self.btn_load_data      = QPushButton("Load Data")

        save_load_buttons = [self.btn_save_config, self.btn_load_config, self.btn_save_data, self.btn_load_data]
        for btn in save_load_buttons:
            btn.setMinimumHeight(32)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E0E0E0;
                    color: #333;
                    border: none;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size:  13px
                }
                QPushButton:hover {
                    background-color: #C0C0C0;
                }
                QPushButton:pressed {
                    background-color: #A0A0A0;
                }
                QPushButton:disabled {
                    background-color: #909090;
                    color: #F0F0F0;
                }
            """)

        button_grid.addWidget(self.btn_save_config, 0, 0)  # First row, first column
        button_grid.addWidget(self.btn_load_config, 0, 1)  # First row, second column
        button_grid.addWidget(self.btn_save_data, 1, 0)    # Second row, first column
        button_grid.addWidget(self.btn_load_data, 1, 1)    # Second row, second column

        button_grid.setSpacing(10)  # 10 pixels spacing between widgets
        button_grid.setContentsMargins(5, 5, 5, 5)  # Margins around the grid

        param_layout.addLayout(button_grid)

        scroll_area.setWidget(param_container)
        return scroll_area
    def _init_middle_panel(self):
        middle_widget = QWidget()
        middle_widget.setMinimumWidth(500)
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(3, 0, 3, 0)     # 左、上、右、下
        middle_layout.setSpacing(2)

        # 创建分割器（垂直方向）
        splitter = QSplitter(Qt.Orientation.Vertical)
        # splitter.setChildrenCollapsible(False)          # 防止组件被完全折叠
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #c0c0c0;
                height: 1px;
            }
        """)

        '''内容显示'''
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumHeight(200)
        self.image_tabs = {}  # 保存 tab 名称到 QLabel 的映射

        # 图像内容
        tab_image_names = [
            "Origin", "PreProcessing", "Spectrum", "Reconstruction", "Focusing",
            "Segmentation", "Identification", "Phase", "Polarization"]
        for name in tab_image_names:
            viewer = Image_Viewer()
            self.image_tabs[name] = viewer
            self.tab_widget.addTab(viewer, name)

        # 数据内容
        tab_data_names  = ["Analysis"]
        for name in tab_data_names:
            viewer = Data_Viewer()
            self.image_tabs[name] = viewer
            self.tab_widget.addTab(viewer, name)

        # 相机画面内容
        tab_camera      = ["Camera"]
        for name in tab_camera:
            camera_label = QLabel('Camera is Not Available (Tab Under Construction ... ...)')
            camera_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            camera_label.setStyleSheet("font-family: Consolas; font-size: 18px; color: blue;font-weight: bold;")
            self.tab_widget.addTab(camera_label, name)

        # 调试页面
        # tab_debug       = ["DeBug"]
        # for name in tab_debug:
        #     viewer = Data_Viewer()
        #     self.image_tabs[name] = viewer
        #     self.tab_widget.addTab(viewer, name)

        self.tab_widget.tabBar().setStyleSheet("""
            QTabBar::tab {
                font-size: 10pt;
                qproperty-alignment: 'AlignCenter';
                padding: 5px 12px;
                font-weight: bold;
                color: #333333;
                background: lightgray;
                border-radius: 2px;
                margin: 1px;
            }
            QTabBar::tab:selected {
                color: white;
                background: #0078d7;
            }
        """)

        '''日志输出'''
        self.log_output = QTextEdit()
        self.log_output.setMinimumHeight(50)
        self.log_output.setStyleSheet("""
            QTextEdit {
                color       : #000000;  /* #5f5f5f */
                font-family : Consolas;
                font-size   : 14px;
                font-weight : bold;
            }
        """)

        # 将组件添加到分割器
        splitter.addWidget(self.tab_widget)
        splitter.addWidget(self.log_output)

        # 设置初始比例（5:1 的比例）
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        middle_layout.addWidget(splitter)

        return middle_widget

    # 这个不要了，单独一个 Widget，用于single mode
    def _init_right_panel__(self):
        panel = QWidget()
        # panel.setFixedWidth(150)
        panel.setMinimumWidth(180)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 0, 3, 0)     # 左、上、右、下

        self.btn_refresh_image  = QPushButton("Refresh Image")      # 刷新Tab图片按钮

        self.btn_operation      = QPushButton("Operation")          # 仅仅用于显示：操作按钮

        self.btn_open_image     = QPushButton("Open Image")
        self.btn_preprocessing  = QPushButton("PreProcessing")
        self.btn_spectrum       = QPushButton("Spectrum")
        self.btn_reconstruction = QPushButton("Reconstruction")
        self.btn_focusing       = QPushButton("Focusing")
        self.btn_segmentation   = QPushButton("Segmentation")
        self.btn_identification = QPushButton("Identification")
        self.btn_phase          = QPushButton("Phase Analysis")
        self.btn_polarization   = QPushButton("Polarization")
        self.btn_data_summary   = QPushButton("Data Summary")
        self.btn_all_in_one     = QPushButton("All in One")

        #移动到左侧菜单栏了
        # self.btn_save_config    = QPushButton("Save Config")
        # self.btn_load_config    = QPushButton("Load Config")
        # self.btn_save_data      = QPushButton("Save Data")
        # self.btn_load_data      = QPushButton("Load Data")

        '''分组_____________________________________'''
        operation_buttons   = [self.btn_open_image, self.btn_preprocessing, self.btn_spectrum, self.btn_reconstruction,
                               self.btn_focusing, self.btn_segmentation, self.btn_identification, self.btn_phase,
                               self.btn_polarization, self.btn_data_summary, self.btn_all_in_one]
        for btn in operation_buttons:
            btn.clicked.connect(lambda _, b=btn: self.log_on_button_clicked(b.text()))
            btn.setMinimumHeight(32)    # 32
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078D7;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #005A9E;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #909090;
                    color: #F0F0F0;
                }
            """)

            if btn.text() == "All in One":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0D47A1;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        padding: 6px 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #1954D2;  /* 1976D2 */
                    }
                    QPushButton:pressed {
                        background-color: #005A9E;
                    }
                    QPushButton:disabled {
                        background-color: #909090;
                        color: #F0F0F0;
                    }
                """)

            if btn.text() == "Data Summary":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #d3d3d3;
                        color       : #333;
                        border      : none;
                        border-radius: 5px;
                        padding     : 6px 12px;
                        font-weight : bold;
                        font-size   : 13px
                    }
                    QPushButton:hover {
                        background-color: #C0C0C0;
                    }
                    QPushButton:pressed {
                        background-color: #A0A0A0;
                    }
                    QPushButton:disabled {
                        background-color: #909090;
                        color: #F0F0F0;
                    }
                """)

        # save_load_buttons   = [self.btn_save_config, self.btn_load_config, self.btn_save_data, self.btn_load_data]
        # for btn in save_load_buttons:
        #     btn.clicked.connect(lambda _, b=btn: self.log_on_button_clicked(b.text()))

        '''布局_____________________________________'''
        '刷新图片'
        self.btn_refresh_image.clicked.connect(lambda: self.log_on_button_clicked(self.btn_refresh_image.text()))
        self.btn_refresh_image.setMinimumHeight(32)
        layout.addWidget(self.btn_refresh_image)
        self.btn_refresh_image.setStyleSheet("""
            QPushButton {
                background-color: #d3d3d3;
                color       : #333;
                border      : none;
                border-radius: 5px;
                padding     : 6px 12px;
                font-weight : bold;
                font-size   : 13px
            }
            QPushButton:hover {
                background-color: #C0C0C0;
            }
            QPushButton:pressed {
                background-color: #A0A0A0;
            }
            QPushButton:disabled {
                background-color: #909090;
                color: #F0F0F0;
            }
        """)

        layout.addStretch(1)

        '标题按钮，仅仅用于提示 Operation 区域'
        self.btn_operation.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;  /* `#101010` 背景色 */
                color: #000000;
                border: none;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        self.btn_operation.setMinimumHeight(25)  # 32
        layout.addWidget(self.btn_operation)

        '操作按钮'
        for btn in operation_buttons:
            layout.addWidget(btn)

        'All in One 按钮的 操作选择'
        layout.addWidget(self.create_operation_options())

        layout.addStretch(3)

        '进度条'
        self.progress_bar_single_operation = QProgressBar()
        self.progress_bar_single_operation.setFixedHeight(16)
        self.progress_bar_single_operation.setTextVisible(False)   # 如果不想显示百分比
        self.progress_bar_single_operation.setRange(0, 100)        # 你可以根据需要设置范围
        self.progress_bar_single_operation.setStyleSheet("""
            QProgressBar {
                border: none;                /* 完全去除边框 */
                background-color: #e0e0e0;   /* 背景色 */
                border-radius: 8px;          /* 圆角 */
                padding: 0px;                /* 去除内边距 */
            }

            QProgressBar::chunk {
                background-color: #4CAF50;   /* 进度条颜色 */
                border-radius: 8px;          /* 与背景相同的圆角 */
                width: 10px;                 /* 块状效果（可选） */
            }
        """)
        self._progress_timer_single_operation = QTimer()
        self._progress_timer_single_operation.timeout.connect(self._update_progress_single_operation)
        layout.addWidget(self.progress_bar_single_operation)

        '计时器标签'
        self.timer_label_single_operation = QLabel("00:00.00 | MM:SS.mm")
        self.timer_label_single_operation.setStyleSheet("font-family: Consolas; font-size: 13px; color: black;font-weight: bold;")
        self.timer_label_single_operation.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self._timer_single_operation = QTimer()
        self._timer_single_operation.timeout.connect(self._update_time_label_single_operation)
        layout.addWidget(self.timer_label_single_operation)

        # 记录按钮
        # layout.addStretch(1)
        # for btn in save_load_buttons:
        #     btn.setMinimumHeight(32)
        #     btn.setStyleSheet("""
        #         QPushButton {
        #             background-color: #E0E0E0;
        #             color: #333;
        #             border: none;
        #             border-radius: 5px;
        #             padding: 6px 12px;
        #         }
        #         QPushButton:hover {
        #             background-color: #C0C0C0;
        #         }
        #     """)
        #     layout.addWidget(btn)

        return panel

    # TabWidget 切换single，multi，camera-realtime模式
    def _init_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(195)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode_tab_widget = QTabWidget(panel)
        # self.mode_tab_widget.setTabPosition(QTabWidget.TabPosition.South)  # 标签页放在下方，也可用 North/West/East
        self.mode_tab_widget.setDocumentMode(True)
        self.mode_tab_widget.setUsesScrollButtons(False)        # 去掉标签切换箭头

        tab_bar = self.mode_tab_widget.tabBar()
        tab_bar.setExpanding(True)  # 让所有 tab 等宽填满

        self.mode_tab_single = self._init_right_panel_single_mode()
        self.mode_tab_multi  = self._init_right_panel_multi_mode()
        self.mode_tab_camera = self._init_right_panel_camera_mode()

        self.mode_tab_widget.addTab(self.mode_tab_single, "Single Image Process")
        self.mode_tab_widget.addTab(self.mode_tab_multi,  "Multi Images Process")
        self.mode_tab_widget.addTab(self.mode_tab_camera, "Camera Real-Time")

        # self.mode_tab_widget.tabBar().setVisible(False)
        self.mode_tab_widget.setTabVisible(1, False)
        self.mode_tab_widget.setTabVisible(2, False)

        self.mode_tab_widget.tabBar().setStyleSheet("""
            QTabBar::tab {
                font-size: 10pt;
                qproperty-alignment: 'AlignCenter';
                padding: 5px 5px;
                font-weight: bold;
                color: #333333;
                background: lightgray;
                border-radius: 2px;
                margin: 1px;
            }
            QTabBar::tab:selected {
                color: white;
                background: #626262;
            }
        """)

        layout.addWidget(self.mode_tab_widget)
        return panel
    def _init_right_panel_single_mode(self):
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_refresh_image  = QPushButton("Refresh Image")
        self.btn_operation      = QPushButton("Operation")

        self.btn_open_image     = QPushButton("Open Image")
        self.btn_preprocessing  = QPushButton("PreProcessing")
        self.btn_spectrum       = QPushButton("Spectrum")
        self.btn_reconstruction = QPushButton("Reconstruction")
        self.btn_focusing       = QPushButton("Focusing")
        self.btn_segmentation   = QPushButton("Segmentation")
        self.btn_identification = QPushButton("Identification")
        self.btn_phase          = QPushButton("Phase Analysis")
        self.btn_polarization   = QPushButton("Polarization")
        self.btn_data_summary   = QPushButton("Data Summary")
        self.btn_all_in_one     = QPushButton("All in One")

        '''分组_____________________________________'''
        operation_buttons = [self.btn_open_image, self.btn_preprocessing, self.btn_spectrum,
                             self.btn_reconstruction, self.btn_focusing, self.btn_segmentation,
                             self.btn_identification, self.btn_phase, self.btn_polarization,
                             self.btn_data_summary, self.btn_all_in_one]
        for btn in operation_buttons:
            btn.clicked.connect(lambda _, b=btn: self.log_on_button_clicked(b.text()))
            btn.setMinimumHeight(30)
            btn.setStyleSheet(self._style_button(btn.text()))

        '''布局_____________________________________'''
        '刷新图片'
        layout.addStretch(1)
        self.btn_refresh_image.clicked.connect(lambda: self.log_on_button_clicked(self.btn_refresh_image.text()))
        self.btn_refresh_image.setMinimumHeight(30)
        self.btn_refresh_image.setStyleSheet(self._style_button("Data Summary"))
        layout.addWidget(self.btn_refresh_image)

        layout.addStretch(1)

        '标题按钮，仅仅用于提示 Operation 区域'
        # layout.addWidget(self.btn_operation)
        # self.btn_operation.setStyleSheet("""
        #     QPushButton {
        #         background-color: #f0f0f0;  /* `#101010` 背景色 */
        #         color: #000000;
        #         border: none;
        #         border-radius: 5px;
        #         padding: 6px 12px;
        #         font-weight: bold;
        #         font-size: 14px;
        #     }
        # """)
        # self.btn_operation.setMinimumHeight(25)  # 32

        '操作按钮'
        for btn in operation_buttons:
            layout.addWidget(btn)

        'All in One 按钮的 操作选择'
        layout.addWidget(self.create_operation_options())

        layout.addStretch(3)

        '进度条'
        self.progress_bar_single_operation = QProgressBar()
        self.progress_bar_single_operation.setFixedHeight(16)
        self.progress_bar_single_operation.setTextVisible(False)
        self.progress_bar_single_operation.setRange(0, 100)
        self.progress_bar_single_operation.setStyleSheet(self._style_progress_bar())
        self._progress_timer_single_operation = QTimer()
        self._progress_timer_single_operation.timeout.connect(self._update_progress_single_operation)
        layout.addWidget(self.progress_bar_single_operation)

        '计时器标签'
        self.timer_label_single_operation = QLabel("00:00.00 | MM:SS.mm")
        self.timer_label_single_operation.setStyleSheet( "font-family: Consolas; font-size: 13px; color: black; font-weight: bold;")
        self.timer_label_single_operation.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self._timer_single_operation = QTimer()
        self._timer_single_operation.timeout.connect(self._update_time_label_single_operation)
        layout.addWidget(self.timer_label_single_operation)

        return panel
    def _init_right_panel_multi_mode(self):
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.create_multi_processing_settings())
        layout.addStretch(5)

        '''计数器标签'''
        self.multi_images_total_num   = 0
        self.multi_images_done_num    = 0
        self.images_num_label = QLabel(f"{self.multi_images_done_num} / {self.multi_images_total_num}")
        self.images_num_label.setStyleSheet("font-family: Consolas; font-size: 13px; color: black; font-weight: bold;")
        self.images_num_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.images_num_label)

        '''进度条'''
        self.progress_bar_multi_images = QProgressBar()
        self.progress_bar_multi_images.setFixedHeight(16)
        self.progress_bar_multi_images.setTextVisible(False)
        self.progress_bar_multi_images.setRange(0, 100)
        self.progress_bar_multi_images.setStyleSheet(self._style_progress_bar())
        self._progress_timer_multi_images = QTimer()
        self._progress_timer_multi_images.timeout.connect(self._update_progress_multi_images)
        layout.addWidget(self.progress_bar_multi_images)

        '''计时器标签'''
        self.timer_label_multi_images = QLabel(f"00:00.00 | MM:SS.mm")
        self.timer_label_multi_images.setStyleSheet("font-family: Consolas; font-size: 13px; color: black; font-weight: bold;")
        self.timer_label_multi_images.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self._timer_multi_images = QTimer()
        self._timer_multi_images.timeout.connect(self._update_time_label_multi_images)
        layout.addWidget(self.timer_label_multi_images)

        layout.addStretch(1)

        '''开始/结束按钮'''
        self.btn_multi_images_start = QPushButton("Start")
        self.btn_multi_images_start.clicked.connect(lambda: self.log_on_button_clicked("Multi-Image Processing Start"))
        self.btn_multi_images_start.setMinimumHeight(32)
        self.btn_multi_images_start.setStyleSheet(self._style_button("Multi-Image Processing Start"))
        layout.addWidget(self.btn_multi_images_start)

        self.btn_multi_images_stop = QPushButton("Stop")
        self.btn_multi_images_stop.setToolTip("Stop After Current Image Finished")  # 这行就是提示设置
        # self.btn_multi_images_stop.clicked.connect(lambda: self.log_on_button_clicked("Multi-Image Processing Stop"))
        self.btn_multi_images_stop.setMinimumHeight(32)
        self.btn_multi_images_stop.setStyleSheet(self._style_button("Multi-Image Processing Stop"))
        layout.addWidget(self.btn_multi_images_stop)

        return panel
    def _init_right_panel_camera_mode(self):
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        label  = QLabel("Under Construction")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return panel

    def _style_button(self, name):
        if   name == "All in One":
            return """
                QPushButton {
                    background-color: #0D47A1;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1954D2;
                }
                QPushButton:pressed {
                    background-color: #005A9E;
                }
                QPushButton:disabled {
                    background-color: #909090;
                    color: #F0F0F0;
                }
            """

        elif name == "Data Summary" or name == "Refresh Image":
            return """
                QPushButton {
                    background-color: #d3d3d3;
                    color       : #333;
                    border      : none;
                    border-radius: 5px;
                    padding     : 6px 12px;
                    font-weight : bold;
                    font-size   : 13px
                }
                QPushButton:hover {
                    background-color: #C0C0C0;
                }
                QPushButton:pressed {
                    background-color: #A0A0A0;
                }
                QPushButton:disabled {
                    background-color: #909090;
                    color: #F0F0F0;
                }
            """

        elif name == "Multi-Image Processing Start":
            return """
                    QPushButton {
                        background-color: #4caf50;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        padding: 6px 12px;
                        font-weight: bold;
                        font-size: 13px;
                        min-width: 80px;
                    }

                    /* 悬停时深绿色 */
                    QPushButton:hover {
                        background-color: #2E7D32;
                    }

                    /* 按下时暗绿色 */
                    QPushButton:pressed {
                        background-color: #1B5E20;
                    }

                    /* 禁用时灰色 */
                    QPushButton:disabled {
                        background-color: #B0BEC5;
                        color: #ECEFF1;
                    }
            """

        elif name == "Multi-Image Processing Stop":
            return """
                    QPushButton {
                        background-color: #E53935;  /* 主红色 */
                        color: white;
                        border: none;
                        border-radius: 5px;
                        padding: 6px 12px;
                        font-weight: bold;
                        font-size: 13px;
                        min-width: 80px;  /* 确保按钮宽度足够 */
                    }

                    /* 悬停时深红色 */
                    QPushButton:hover {
                        background-color: #C62828;
                    }

                    /* 按下时暗红色 */
                    QPushButton:pressed {
                        background-color: #B71C1C;
                    }

                    /* 禁用时灰色 */
                    QPushButton:disabled {
                        background-color: #B0BEC5;
                        color: #ECEFF1;
                    }
            """

        else:
            return """
                QPushButton {
                    background-color: #0078D7;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #005A9E;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #909090;
                    color: #F0F0F0;
                }
            """
    def _style_progress_bar(self):
        return """
            QProgressBar {
                border: none;
                background-color: #e0e0e0;
                border-radius: 8px;
                padding: 0px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 8px;
                width: 10px;
            }
        """

    '''参数设置菜单组件'''
    def create_file_info_group(self):
        # conf = self.config.get("file_info", {})
        conf = self.config.file_info if hasattr(self.config, 'file_info') else {}
        return self._create_group("File Info", {
            "file_info.holo_type": {
                "type"   : "combo",
                "value"  : conf.get("holo_type", "Inline"),
                "options": conf.get("holo_type_list", []),
                "name"   : "Hologram Type"
            },
            "file_info.image_path": {
                "type"   : "input",
                "value"  : conf.get("image_path", ""),
                "name"   : "Image Path"
            },
            "file_info.image_name": {
                "type": "input",
                "value": conf.get("image_name", ""),
                "name": "Image Name"
            },
            # "line1": {"type": "line"},
            # "file_info.read_mode": {
            #     "type": "combo",
            #     "value": conf.get("read_mode", "None"),
            #     "options": conf.get("read_mode_list", []),
            #     "name": "Read Mode"
            # },
        })
    def create_image_info_group(self):
        # conf = self.config.get("image_info", {})
        conf = self.config.image_info if hasattr(self.config, 'image_info') else {}
        roi  = conf.get("ROI_rectangle", {})
        return self._create_group("Image Info", {
            "image_info.pixel_size": {"type": "input", "value": str(conf.get("pixel_size", "5.0")), "name": "Pixel Size (um)"},
            "image_info.wavelength": {"type": "input", "value": str(conf.get("wavelength", "532.0")), "name": "Wavelength (nm)"},
            "image_info.pixel_num_x": {"type": "input", "value": str(conf.get("pixel_num_x", "1024")), "name": "Pixel Num X"},
            "image_info.pixel_num_y": {"type": "input", "value": str(conf.get("pixel_num_y", "1024")), "name": "Pixel Num Y"},
            "image_info.off_axis_angle_x": {"type": "input", "value": str(conf.get("off_axis_angle_x", "0.0")), "name": "Off Angle X (deg)"},
            "image_info.off_axis_angle_y": {"type": "input", "value": str(conf.get("off_axis_angle_y", "0.0")), "name": "Off Angle Y (deg)"},
            "image_info.ROI_rectangle.center_x": {"type": "input", "value": str(roi.get("center_x", "50")), "name": "ROI Center X"},
            "image_info.ROI_rectangle.center_y": {"type": "input", "value": str(roi.get("center_y", "50")), "name": "ROI Center Y"},
            "image_info.ROI_rectangle.rect_width": {"type": "input", "value": str(roi.get("rect_width", "50")), "name": "ROI Width"},
            "image_info.ROI_rectangle.rect_height": {"type": "input", "value": str(roi.get("rect_height", "50")), "name": "ROI Height"}
        })
    def create_preprocess_group(self):
        # conf = self.config.get("pre_process", {})
        conf = self.config.pre_process if hasattr(self.config, 'pre_process') else {}

        return self._create_group("Preprocess", {
            "pre_process.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "pre_process.coeff": {"type": "input", "value": str(conf.get("coeff", "0.5")), "name": "Coefficient"},
            "pre_process.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "pre_process.background_path": {"type": "input", "value": conf.get("background_path", ""), "name": "Background Path"},
            "pre_process.background_name": {"type": "input", "value": conf.get("background_name", ""), "name": "Background Name"},
            "pre_process.model_path": {
                "type": "input",
                "value": conf.get("model_path", ""),
                "name": "Model Path"
            },
            "pre_process.model_name": {
                "type": "input",
                "value": conf.get("model_name", ""),
                "name": "Model Name"
            }
        })
    def create_spectrum_group(self):
        # conf = self.config.get("spectrum", {})
        conf = self.config.spectrum if hasattr(self.config,'spectrum') else {}
        roi  = conf.get("ROI_rectangle", {})
        return self._create_group("Spectrum", {
            "spectrum.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "spectrum.center_mask_radius": {
                "type": "input",
                "value": str(conf.get("center_mask_radius", "")),
                "name": "Center Mask Radius"
            },

            "spectrum.threshold": {
                "type": "input",
                "value": str(conf.get("threshold", "")),
                "name": "Threshold"
            },

            "spectrum.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "spectrum.ROI_rectangle.center_x": {"type": "input", "value": str(roi.get("center_x", "50")), "name": "ROI Center X"},
            "spectrum.ROI_rectangle.center_y": {"type": "input", "value": str(roi.get("center_y", "50")), "name": "ROI Center Y"},
            "spectrum.ROI_rectangle.rect_width": {"type": "input", "value": str(roi.get("rect_width", "50")), "name": "ROI Width"},
            "spectrum.ROI_rectangle.rect_height": {"type": "input", "value": str(roi.get("rect_height", "50")), "name": "ROI Height"}
        })
    def create_reconstruction_group(self):
        # conf = self.config.get("reconstruction", {})
        conf = self.config.reconstruction if hasattr(self.config,'reconstruction') else {}
        return self._create_group("Reconstruction", {
            "reconstruction.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "reconstruction.z_start": {"type": "input", "value": str(conf.get("z_start", "0.0")), "name": "Z Start (mm)"},
            "reconstruction.z_end": {"type": "input", "value": str(conf.get("z_end", "10.0")), "name": "Z End (mm)"},
            "reconstruction.z_step": {"type": "input", "value": str(conf.get("z_step", "1.0")), "name": "Z Step (mm)"},
            "line1": {"type": "line"},
            "reconstruction.cpu_num": {"type": "input", "value": str(conf.get("cpu_num", "1")), "name": "CPU Num"},
            "reconstruction.gpu_num": {"type": "input", "value": str(conf.get("gpu_num", "1")), "name": "GPU Num"},
            "reconstruction.model_path": {
                "type": "input",
                "value": conf.get("model_path", ""),
                "name": "Model Path"
            },
            "reconstruction.model_name": {
                "type": "input",
                "value": conf.get("model_name", ""),
                "name": "Model Name"
            }
        })

    def create_focusing_group_(self):
        # conf = self.config.get("focusing", {})
        conf = self.config.focusing if hasattr(self.config, 'focusing') else {}
        return self._create_group("Focusing", {
            "focusing.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "focusing.cpu_num": {"type": "input", "value": str(conf.get("cpu_num", "1")), "name": "CPU Num"},
            "focusing.gpu_num": {"type": "input", "value": str(conf.get("gpu_num", "1")), "name": "GPU Num"},
            "focusing.model_path": {
                "type": "input",
                "value": conf.get("model_path", ""),
                "name": "Model Path"
            },
            "focusing.model_name": {
                "type": "input",
                "value": conf.get("model_name", ""),
                "name": "Model Name"
            }
        })
    def create_focusing_group(self):
        # conf = self.config.get("focusing", {})
        conf = self.config.focusing if hasattr(self.config, 'focusing') else {}
        return self._create_group("Focusing", {
            "focusing.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },

            "focusing.yolo_model_path": {
                "type": "input",
                "value": conf.get("yolo_model_path", ""),
                "name": "Yolo Path"
            },
            "focusing.rcf_model_path": {
                "type": "input",
                "value": conf.get("rcf_model_path", ""),
                "name": "RCF Path"
            },
            "focusing.rcf_scale": {
                "type": "input",
                "value": str(conf.get("rcf_scale", "8")),
                "name": "RCF Scale"
            },

            "line1": {"type": "line"},

            "focusing.device": {
                "type": "combo",
                "value": conf.get("device", "cpu"),
                "options": conf.get("device_list", []),
                "name": "Device"
            },
            "focusing.cpu_num": {
                "type": "input",
                "value": str(conf.get("cpu_num", "1")),
                "name": "CPU Num"},
            "focusing.gpu_id": {
                "type": "input",
                "value": str(conf.get("gpu_id", "0")),
                "name": "GPU ID"
            },

            "line2": {"type": "line"},

            "focusing.get_model": {
                "type": "checkbox",
                "value": conf.get("get_model", False),
                "name": "Get Model by Function (For Bacth)"
            },

        })

    def create_segmentation_group_(self):
        # conf = self.config.get("segmentation", {})
        conf = self.config.segmentation if hasattr(self.config,'segmentation') else {}
        return self._create_group("Segmentation", {
            "segmentation.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "segmentation.gray_thresh": {"type": "input", "value": str(conf.get("gray_thresh", "127")), "name": "Gray Threshold"},
            "segmentation.block_size": {"type": "input", "value": str(conf.get("block_size", "32")), "name": "Block Size"},
            "segmentation.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "segmentation.model_name": {"type": "input", "value": conf.get("model_name", ""), "name": "Model Name"},
        })
    def create_segmentation_group(self):
        # conf = self.config.get("segmentation", {})
        conf = self.config.segmentation if hasattr(self.config,'segmentation') else {}
        return self._create_group("Segmentation", {
            "segmentation.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "segmentation.type": {
                "type": "combo",
                "value": conf.get("type", "None"),
                "options": conf.get("type_list", []),
                "name": "Image Type"
            },
            "line1": {"type": "line"},
            "segmentation.device": {
                "type": "combo",
                "value": conf.get("device", "cpu"),
                "options": conf.get("device_list", []),
                "name": "Device"
            },
            "segmentation.cpu_num": {
                "type": "input",
                "value": str(conf.get("cpu_num", "1")),
                "name": "CPU Num"},
            "segmentation.gpu_num": {
                "type": "input",
                "value": str(conf.get("gpu_num", "0")),
                "name": "GPU Num"
            },

            "line2": {"type": "line"},
            "segmentation.gray_thresh": {
                "type": "input",
                "value": str(conf.get("gray_thresh", "127")),
                "name": "Gray Threshold"
            },
            "segmentation.block_size": {
                "type": "input",
                "value": str(conf.get("block_size", "32")),
                "name": "Block Size"
            },

            "line3": {"type": "line"},
            "segmentation.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "segmentation.model_name": {"type": "input", "value": conf.get("model_name", ""), "name": "Model Name"},
        })

    def create_identification_group_(self):
        # conf = self.config.get("identification", {})
        conf = self.config.identification if hasattr(self.config, 'identification') else {}
        return self._create_group("Identification", {
            "identification.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "identification.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "identification.model_name": {"type": "input", "value": conf.get("model_name", ""), "name": "Model Name"},
        })
    def create_identification_group(self):
        # conf = self.config.get("identification", {})
        conf = self.config.identification if hasattr(self.config, 'identification') else {}
        return self._create_group("Identification", {
            "identification.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },

            "identification.type": {
                "type": "combo",
                "value": conf.get("type", "None"),
                "options": conf.get("type_dict", []),
                "name": "Particle Type"
            },

            "identification.model_path": {
                "type": "input",
                "value": conf.get("model_path", ""),
                "name": "Model Path"
            },
            "identification.model_name": {
                "type": "input",
                "value": conf.get("model_name", ""),
                "name": "Model Name"},
        })

    def create_phase_group(self):
        # conf = self.config.get("phase", {})
        conf = self.config.phase if hasattr(self.config, 'phase') else {}
        return self._create_group("Phase Analysis", {
            "phase.method": {
                "type": "combo",
                "value": conf.get("method", "None"),
                "options": conf.get("method_list", []),
                "name": "Method"
            },
            "phase.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "phase.model_name": {"type": "input", "value": conf.get("model_name", ""), "name": "Model Name"}
        })
    def create_polarization_group(self):
        conf = self.config.polarization if hasattr(self.config, 'polarization') else {}
        return self._create_group("Polarization", {
            "polarization.split_image": {
                "type": "checkbox",
                "value": conf.get("split_image", True),
                "name": "Split Polarization Image"
            },

            "polarization.split_mode": {
                "type": "combo",
                "value": conf.get("split_mode", "None"),
                "options": conf.get("split_mode_list", []),
                "name": "Split Mode"
            },

            "line1": {"type": "line"},

            "polarization.polar_coeff": {
                "type": "checkbox",
                "value": conf.get("polar_coeff", True),
                "name": "Calculate Polarization Parameters"
            },

            "polarization.device": {
                "type": "combo",
                "value": conf.get("device", "cpu"),
                "options": conf.get("device_list", []),
                "name": "Device"
            }
        })

    def create_data_summary_group(self):
        # conf = self.config.get("data_summary", {})
        conf = self.config.data_summary if hasattr(self.config, 'data_summary') else {}
        return self._create_group("Data Analysis", {
            "data_summary.density": {"type": "input", "value": str(conf.get("density", "")), "name": "Density"},
            "data_summary.model_path": {"type": "input", "value": conf.get("model_path", ""), "name": "Model Path"},
            "data_summary.model_name": {"type": "input", "value": conf.get("model_name", ""), "name": "Model Name"},
            "data_summary.show_size_distribution": {
                "type": "checkbox",
                "value": conf.get("show_size_distribution", True),
                "name": "Size Distribution"
            },
            "data_summary.show_concentration_time": {
                "type": "checkbox",
                "value": conf.get("show_concentration_time", True),
                "name": "Concentration over Time"
            },
            "data_summary.show_R50_distribution": {
                "type": "checkbox",
                "value": conf.get("show_R50_distribution", True),
                "name": "R50 Distribution"
            },
            "data_summary.show_R90_distribution": {
                "type": "checkbox",
                "value": conf.get("show_R90_distribution", True),
                "name": "R90 Distribution"
            },
            "data_summary.show_R200_distribution": {
                "type": "checkbox",
                "value": conf.get("show_R200_distribution", True),
                "name": "R200 Distribution"
            }

        })
    def create_save_and_load_group(self):
        conf = self.config.save_and_load if hasattr(self.config, 'save_and_load') else {}
        return self._create_group("Save | Load", {
            "group1": {"type": "group", "name": "Config Load/Save"},

            "save_and_load.config_save_path": {
                "type": "input",
                "value": conf.get("config_save_path", ""),
                "name": "Save Path"
            },
            "save_and_load.config_save_name": {
                "type": "input",
                "value": conf.get("config_save_name", ""),
                "name": "Save Name"
            },
            "save_and_load.config_load_path": {
                "type": "input",
                "value": conf.get("config_load_path", ""),
                "name": "Load Path"
            },
            "save_and_load.config_load_name": {
                "type": "input",
                "value": conf.get("config_load_name", ""),
                "name": "Load Name"
            },

            "group2": {"type": "group", "name": "Data Load/Save"},

            "save_and_load.data_load_path": {
                "type": "input",
                "value": conf.get("data_load_path", ""),
                "name": "Load Path"
            },
            "save_and_load.data_load_name":{
                "type": "input",
                "value": conf.get("data_load_name", ""),
                "name": "Load Name"
            },
            # "save_and_load.data_NPZ_only": {
            #     "type": "checkbox",
            #     "value": conf.get("data_NPZ_only", True),
            #     "name": "Load NPZ Data Only"
            # },

            "line1": {"type": "line"},

            "save_and_load.data_save_method": {
                "type": "combo",
                "value": conf.get("data_save_method", "None"),
                "options": conf.get("data_save_method_list", []),
                "name": "Save Method"
            },
            "save_and_load.data_save_path": {
                "type": "input",
                "value": conf.get("data_save_path", ""),
                "name": "Save Path"
            },
            "save_and_load.data_save_name": {
                "type": "input",
                "value": conf.get("data_save_name", ""),
                "name": "Save Name"
            },

            "group3": {"type": "group", "name": "Option for Save"},

            "save_and_load.save_to_disk": {
                "type": "checkbox",
                "value": conf.get("save_to_disk", True),
                "name": "Auto Save Resulted Image When Processed"
            },
            "save_and_load.creat_sub_dir": {
                "type": "checkbox",
                "value": conf.get("creat_sub_dir", True),
                "name": "Creat Sub Dir by Image Name"
            },

            "line2": {"type": "line"},

            "save_and_load.save_all": {
                "type": "checkbox",
                "value": conf.get("save_all", True),
                "name": "Save All"
            },
            "save_and_load.save_raw_hologram": {
                "type": "checkbox",
                "value": conf.get("save_raw_hologram", True),
                "name": "Save Raw Hologram"
            },
            "save_and_load.save_pre_process": {
                "type": "checkbox",
                "value": conf.get("save_pre_process", True),
                "name": "Save Pre-Processed Hologram"
            },
            "save_and_load.save_polarization": {
                "type": "checkbox",
                "value": conf.get("save_polarization", True),
                "name": "Save Polarization Hologram"
            },
            "save_and_load.save_spectrum": {
                "type": "checkbox",
                "value": conf.get("save_spectrum", True),
                "name": "Save Spectrum"
            },
            "save_and_load.save_reconstruction": {
                "type": "checkbox",
                "value": conf.get("save_reconstruction", True),
                "name": "Save Reconstruction"
            },
            "save_and_load.save_focusing": {
                "type": "checkbox",
                "value": conf.get("save_focusing", True),
                "name": "Save Focusing"
            },
            "save_and_load.save_segmentation": {
                "type": "checkbox",
                "value": conf.get("save_segmentation", True),
                "name": "Save Segmentation"
            },
            "save_and_load.save_identification": {
                "type": "checkbox",
                "value": conf.get("save_identification", True),
                "name": "Save Identification"
            },
            "save_and_load.save_phase": {
                "type": "checkbox",
                "value": conf.get("save_phase", True),
                "name": "Save Phase"
            },
            "save_and_load.save_data_summary": {
                "type": "checkbox",
                "value": conf.get("save_data_summary", True),
                "name": "Save Data Analysis"
            }
        })
    def create_operation_options(self):
        conf = self.config.operation_options if hasattr(self.config, 'operation_options') else {}
        return self._create_group_unfolded("Operation Options", {
            "operation_options.run_open_image": {
                "type": "checkbox",
                "value": conf.get("run_open_image", True),
                "name": "Open New Image"
            },
            "operation_options.run_preprocessing": {
                "type": "checkbox",
                "value": conf.get("run_preprocessing", True),
                "name": "Pre-Processing"
            },
            "operation_options.run_polarization": {
                "type": "checkbox",
                "value": conf.get("run_polarization", True),
                "name": "Polarization"
            },
            "operation_options.run_spectrum": {
                "type": "checkbox",
                "value": conf.get("run_spectrum", True),
                "name": "Spectrum"
            },
            "operation_options.run_reconstruction": {
                "type": "checkbox",
                "value": conf.get("run_reconstruction", True),
                "name": "Reconstruction"
            },
            "operation_options.run_focusing": {
                "type": "checkbox",
                "value": conf.get("run_focusing", True),
                "name": "Focusing"
            },
            "operation_options.run_segmentation": {
                "type": "checkbox",
                "value": conf.get("run_segmentation", True),
                "name": "Segmentation"
            },
            "operation_options.run_identification": {
                "type": "checkbox",
                "value": conf.get("run_identification", True),
                "name": "Identification"
            },
            "operation_options.run_phase": {
                "type": "checkbox",
                "value": conf.get("run_phase", True),
                "name": "Phase"
            },
            "operation_options.run_data_summary": {
                "type": "checkbox",
                "value": conf.get("run_data_summary", True),
                "name": "Data Analysis"
            }
        })
    def create_multi_processing_settings(self):
        conf = self.config.multi_processing if hasattr(self.config, 'multi_processing') else {}
        return self._create_group("Multi Processing Settings", {

            "group1": {"type": "group", "name": "Images Folder"},

            "multi_processing.images_path": {
                "type": "input_only",
                "value": conf.get("images_path", ""),
                "name": "Path"
            },

            "multi_processing.images_name_suffix": {
                "type": "combo",
                "value": conf.get("images_name_suffix", "any"),
                "options": conf.get("images_name_suffix_list", []),
                "name": "Name Suffix"
            },

            "multi_processing.max_handle_num": {
                "type": "input",
                "value": str(conf.get("max_handle_num", "None")),
                "name": "Max Num"
            },

            "group2": {"type": "group", "name": "Operation for Each"},

            "multi_processing.run_open_image": {
                "type": "checkbox",
                "value": conf.get("run_open_image", True),
                "name": "Open New Image"
            },
            "multi_processing.run_preprocessing": {
                "type": "checkbox",
                "value": conf.get("run_preprocessing", True),
                "name": "Pre-Processing"
            },
            "multi_processing.run_polarization": {
                "type": "checkbox",
                "value": conf.get("run_polarization", True),
                "name": "Polarization"
            },
            "multi_processing.run_spectrum": {
                "type": "checkbox",
                "value": conf.get("run_spectrum", True),
                "name": "Spectrum"
            },
            "multi_processing.run_reconstruction": {
                "type": "checkbox",
                "value": conf.get("run_reconstruction", True),
                "name": "Reconstruction"
            },
            "multi_processing.run_focusing": {
                "type": "checkbox",
                "value": conf.get("run_focusing", True),
                "name": "Focusing"
            },
            "multi_processing.run_segmentation": {
                "type": "checkbox",
                "value": conf.get("run_segmentation", True),
                "name": "Segmentation"
            },
            "multi_processing.run_identification": {
                "type": "checkbox",
                "value": conf.get("run_identification", True),
                "name": "Identification"
            },
            "multi_processing.run_phase": {
                "type": "checkbox",
                "value": conf.get("run_phase", True),
                "name": "Phase"
            },
            "multi_processing.run_data_summary": {
                "type": "checkbox",
                "value": conf.get("run_data_summary", True),
                "name": "Data Analysis"
            }
        }, expanded=True, show_button=False)

    def _create_group(self, title, items: dict, expanded=False, show_button=True):
        section = Collapsible_Section(title, title_color="#626262", expanded=expanded, show_button=show_button)
        form_layout = QFormLayout()

        for key_path, info in items.items():
            field_type  = info.get("type", "input")
            value       = info.get("value")
            label_name  = info.get("name", key_path.split(".")[-1].replace("_", " ").capitalize())

            if field_type == "checkbox":
                widget = self._add_checkbox(key_path, value, label_name)
                form_layout.addRow(widget)  # Checkbox 自带 label

            elif field_type == "combo":
                options = info.get("options", [])
                self._add_combobox(form_layout, key_path, current=value, options=options, label_text=label_name)

            elif field_type == "input_only":
                self._add_input_field_only(form_layout, key_path, value)

            elif field_type == "group":
                self._add_group_separator(form_layout, label_name)

            elif field_type == "line":
                self._add_line_separator(form_layout)

            else:  # 默认是 label : input
                self._add_label_and_input(form_layout, key_path, value, label_text=label_name)

        container = QWidget()
        container.setLayout(form_layout)
        section.add_widget(container)
        return section
    def _create_group_unfolded(self, title, items: dict):
        form_layout = QFormLayout()
        form_layout.setContentsMargins(10, 0, 0, 0)  # 左、上、右、下
        form_layout.setSpacing(2)                   # 设置控件之间的间距为（默认一般是 6 或更大）

        for key_path, info in items.items():
            field_type  = info.get("type", "input")
            value       = info.get("value")
            label_name  = info.get("name", key_path.split(".")[-1].replace("_", " ").capitalize())

            if field_type == "checkbox":
                widget = self._add_checkbox(key_path, value, label_name)
                form_layout.addRow(widget)  # Checkbox 自带 label

            elif field_type == "combo":
                options = info.get("options", [])
                self._add_combobox(form_layout, key_path, current=value, options=options, label_text=label_name)

            elif field_type == "input_only":
                self._add_label_and_input(form_layout, key_path, value)

            elif field_type == "group":
                self._add_group_separator(form_layout, label_name)

            elif field_type == "line":
                self._add_line_separator(form_layout)

            else: # 默认是 label : input
                self._add_label_and_input(form_layout, key_path, value, label_text=label_name)

        container = QWidget()
        container.setLayout(form_layout)
        return container
    def _add_label_and_input(self, layout, key_path, default, label_text=None, short_label=False):
        label = QLabel(label_text or key_path.split(".")[-1].replace("_", " ").capitalize())
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet("font-weight: bold;")
        label.setFixedWidth(120)

        edit = QLineEdit(default)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        edit.setProperty("key_path", key_path)
        edit.editingFinished.connect(self._on_param_changed)

        layout.addRow(label, edit)
        self.input_fields[key_path] = edit
        return edit
    def _add_input_field_only(self, layout, key_path, default):
        edit = QLineEdit(default)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        edit.setProperty("key_path", key_path)
        edit.editingFinished.connect(self._on_param_changed)

        layout.addRow(edit)
        self.input_fields[key_path] = edit
        return edit
    def _add_checkbox(self, key_path, default, label_text=None):
        checkbox = QCheckBox()
        checkbox.setChecked(default)
        checkbox.stateChanged.connect(lambda state: self.param_changed.emit(key_path, bool(state)))
        checkbox.setObjectName(key_path)

        label = label_text or key_path.split(".")[-1].replace("_", " ").capitalize()
        checkbox.setText(label)
        checkbox.setStyleSheet("font-weight: bold;")

        self.input_fields[key_path] = checkbox
        return checkbox
    def _add_combobox(self, layout, key_path, current, options, label_text=None):
        combo = QComboBox()
        combo.addItems(options)

        if current in options:
            combo.setCurrentText(current)

        combo.setEditable(True)
        combo.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        combo.lineEdit().setReadOnly(True)
        # combo.setStyleSheet(("font-weight: bold;"))

        combo.setProperty("key_path", key_path)
        combo.currentTextChanged.connect(lambda val: self.param_changed.emit(key_path, val))

        label = QLabel(label_text or key_path.split(".")[-1].replace("_", " ").capitalize())
        label.setFixedWidth(120)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet(("font-weight: bold;"))

        layout.addRow(label, combo)
        self.input_fields[key_path] = combo
    def _add_group_separator(self, layout, text: str):
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5) # 左、上、右、下
        container_layout.setSpacing(5)

        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; color: #000000;")
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setLineWidth(2)  # 线宽
        line.setStyleSheet("color: #626262; background-color: #626262;border-top: 10px")
        line.setFixedHeight(1)

        container_layout.addWidget(label)
        container_layout.addWidget(line, stretch=1)

        layout.addRow(container)
    def _add_line_separator(self, layout):
        ''''''
        '''普通'''
        # line = QFrame()
        # line.setFrameShape(QFrame.Shape.HLine)
        # line.setFrameShadow(QFrame.Shadow.Sunken)
        # # line.setLineWidth(2)  # 线宽
        # # line.setMidLineWidth(1)  # 中线宽度（双线效果）
        # layout.addRow(line)

        '''加个容器控制边距'''
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5) # 左、上、右、下

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        # line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setLineWidth(2)  # 线宽
        line.setStyleSheet("color: #a2a2a2; background-color: #a2a2a2;border-top: 10px")
        line.setFixedHeight(1)
        container_layout.addWidget(line, stretch=1)

        layout.addRow(container)

        '''视觉增强'''
        # widget = QWidget()
        # widget.setLayout(QHBoxLayout())
        #
        # # 左侧线
        # left_line = QFrame()
        # left_line.setFrameShape(QFrame.Shape.HLine)
        # left_line.setStyleSheet("border-top: 1px solid #D0D0D0;")
        #
        # # 中间图标（可选）
        # icon = QLabel("⚡")  # 或使用 QPixmap
        # icon.setStyleSheet("color: #606060; padding: 0 5px;")
        #
        # # 右侧线
        # right_line = QFrame()
        # right_line.setFrameShape(QFrame.Shape.HLine)
        # right_line.setStyleSheet("border-top: 1px solid #D0D0D0;")
        #
        # # 添加到布局
        # widget.layout().addWidget(left_line)
        # widget.layout().addWidget(icon)
        # widget.layout().addWidget(right_line)
        # widget.layout().setContentsMargins(0, 0, 0, 0)
        #
        # layout.addRow(widget)

    '''信号 参数/按钮'''
    # 参数修改
    def _on_param_changed(self):
        edit        = self.sender()
        key_path    = edit.property("key_path")
        text        = edit.text()

        # 自动判断类型（float > int > str）
        try:
            if "." in text or "e" in text.lower():
                value = float(text)
            else:
                value = int(text)
        except ValueError:
            value = text

        self.param_changed.emit(key_path, value)
    def log_on_param_changed(self, key, value):
        # msg = f"[Parameter Changed] {key} = {value}"
        # self.log_output.append(msg)
        # self.statusBar().showMessage(msg) # showMessage(msg, 5000)

        # html
        msg_html = f'<span style="color:black;">[Parameter]</span> {key} = {value}'
        self.log_output.append(msg_html)

        # 状态栏保持纯文本
        msg = f"[Parameter Changed] {key} = {value}"
        self.statusBar().showMessage(f"[Parameter Changed] {key} = {value}")

    # 按钮操作 | 计算完毕之后，Tab页面图像更新
    def update_image__(self, operation_name: str, image, status: str):
        ''''''
        '''
        过来的 image 变量有三种情况
        1、None：表示其他动作，保存等，无需计时
        2、空字典{}  ：表示开始处理，会启动计时
        3、非空字典，ndarray：表示处理完成，显示图像，如果计时启动过，则停止计时 | 如果计时没启动过，则不计时
        '''

        status_msg = status
        msg_html   = f'<span style="color:#4caf50;">[Operation] {status_msg}</span>'    # #FF00FF, #00C400
        self.log_output.append(msg_html)
        self.statusBar().showMessage(status_msg)

        '''图像为None → 表示其他动作，保存等，无需计时'''
        if image is None:
            ''''''
            return

        '''图像为空 {} → 表示开始处理，显示进度条动画'''
        if (isinstance(image, dict) and not image):
            self._enable_time_count = True  # 主要用来标记，本次处理需要计时

            # 启动进度条动画
            self.progress_bar_single_operation.setRange(0, 0)  # 不确定进度的“繁忙”状态（动的）
            self._progress_timer_single_operation.start(100)             # 可以根据需要调整速度（毫秒）

            # 启动计时器
            self._operation_seconds = 0
            self.timer_label_single_operation.setText("00:00.00 | MM:SS.mm")
            self._timer_single_operation.start(100)  # 每0.1秒更新一次

            self._button_locked()  # 锁定按钮
            return

        '''图像有效 → 表示处理完成'''
        if self._timer_single_operation.isActive():
            # 停止进度条
            self.progress_bar_single_operation.setRange(0, 100)  # 确定进度的“空闲”状态（静的）
            self._progress_timer_single_operation.stop()

            # 停止计时器
            self._timer_single_operation.stop()

            # 计算时间（浮点数）
            total_seconds = self._operation_seconds  # 假设是浮点数（如 12.345 秒）
            minutes = int(total_seconds // 60)
            seconds_with_ms = total_seconds % 60  # 包含小数的秒数（如 5.678 秒）

            # 格式化显示：MM:SS.mm
            time_str = f"{minutes:02d}:{seconds_with_ms:05.2f} | MM:SS.mm"  # 例如 "00:12.34"
            self.timer_label_single_operation.setText(time_str)

            msg_time_usage = f"Time usage: {total_seconds:.2f} /s"
            self.statusBar().showMessage(f"[Operation] {msg_time_usage}")

            msg_html = f'<span style="color:#777777;">[Operation] {msg_time_usage}</span>'  # #FF00FF, #00C400
            self.log_output.append(msg_html)

            self._enable_time_count = False  # 标记结束

            self._button_released()

        '''显示图像，如果是ndarray，转成字典、如果是字典，则直接用'''
        # 如果是读取图片，则更新全息图像类型
        if operation_name == "Origin":
            holo_type = str(self.config.file_info['holo_type'])
            self.input_fields["file_info.holo_type"].setCurrentText(holo_type)

        if isinstance(image, np.ndarray):
            image_dict = {operation_name: image}
        elif isinstance(image, dict):
            image_dict = image
        else:
            print(type(image))
            print(f"[Warning] Unsupported image type for update_image: {type(image)}")
            return

        viewer = self.image_tabs.get(operation_name)
        if viewer:
            # 自动切换到对应 tab 页面
            for index in range(self.tab_widget.count()):
                if self.tab_widget.widget(index) is viewer:
                    self.tab_widget.setCurrentIndex(index)
                    break

            viewer.set_image(image) # 不管是图片还是数据，都有set_image这个方法
        else:
            print(f"[Warning] No viewer found for tab '{operation_name}'")

    '''
        兼容不同处理模式
        1、单张图
        2、多张图
    
        对于单张图模式，过来的 image 变量有三种情况
        1、None：表示其他动作，保存等，无需计时
        2、空字典{}  ：表示开始处理，会启动计时
        3、非空字典，ndarray：表示处理完成，显示图像，如果计时启动过，则停止计时 | 如果计时没启动过，则不计时
    '''
    def update_image(self, operation_name: str, image, status: str):
        ''''''
        '''Case 1. 多张图像'''
        if operation_name == "Multi Processing":
            self._button_locked()
            self._handle_multi_processing(operation_name, image, status)
            return

        '''Case 2. 单张图像'''
        self._update_image_log(status)

        if image is None:
            return

        if isinstance(image, dict) and not image:
            self._start_operation_timer()
            return

        self._finish_operation_timer()
        self._update_image_viewer(operation_name, image)
    def _update_image_log(self, status: str, color = '#4caf50'):
        # msg_html = f'[Operation] {status}'
        msg_html = f'<span style="color:{color};">[Operation] {status}</span>'  # #FF00FF, #00C400
        self.statusBar().showMessage(status)
        self.log_output.append(msg_html)
    def _update_image_viewer(self, operation_name: str, image_data):
        """根据 operation_name 更新对应的图像展示区域"""

        if operation_name == "Origin":
            self._handle_open_image()

        if operation_name == "Spectrum":
            self._handle_spectrum()

        # 标准化为字典形式
        image_dict = {operation_name: image_data} if isinstance(image_data, np.ndarray) else image_data

        viewer = self.image_tabs.get(operation_name)
        if not viewer:
            print(f"[Warning] No viewer found for tab '{operation_name}'")
            return

        # 自动切换到对应的 图像tab 页面
        for index in range(self.tab_widget.count()):
            if self.tab_widget.widget(index) is viewer:
                self.tab_widget.setCurrentIndex(index)
                break

        viewer.set_image(image_dict)
    def _handle_open_image(self):
        '''主要用来更新 GUI 上的图像尺寸、类型信息'''
        holo_type    = str(self.config.file_info['holo_type'])
        image_height = str(self.config.image_info['pixel_num_x'])
        image_width  = str(self.config.image_info['pixel_num_y'])

        self.input_fields["file_info.holo_type"].setCurrentText(holo_type)
        self.input_fields["image_info.pixel_num_x"].setText(image_height)
        self.input_fields["image_info.pixel_num_y"].setText(image_width)

        '''备份：本来想在前端GUI处理图像的尺寸信息，直接读取，然后计算'''
        # path_field = self.input_fields.get("file_info.image_path")
        # name_field = self.input_fields.get("file_info.image_name")
        #
        # if not path_field or not name_field:
        #     return
        #
        # image_path = path_field.text()
        # image_name = name_field.text()
        # full_path = os.path.join(image_path, image_name)
        #
        # if not os.path.exists(full_path):
        #     return
        #
        # try:
        #     with Image.open(full_path) as img:
        #         width, height = img.size
        #
        #         x_field = self.input_fields.get("image_info.pixel_num_x")
        #         y_field = self.input_fields.get("image_info.pixel_num_y")
        #
        #         if x_field:
        #             x_field.setText(str(width))
        #         if y_field:
        #             y_field.setText(str(height))
        #
        # except Exception as e:
        #     return
    def _handle_spectrum(self):

        center_x    = str(self.config.spectrum['ROI_rectangle']['center_x'])
        center_y    = str(self.config.spectrum['ROI_rectangle']['center_y'])
        rect_width  = str(self.config.spectrum['ROI_rectangle']['rect_width'])
        rect_height = str(self.config.spectrum['ROI_rectangle']['rect_height'])

        self.input_fields["spectrum.ROI_rectangle.center_x"].setText(center_x)
        self.input_fields["spectrum.ROI_rectangle.center_y"].setText(center_y)
        self.input_fields["spectrum.ROI_rectangle.rect_width"].setText(rect_width)
        self.input_fields["spectrum.ROI_rectangle.rect_height"].setText(rect_height)

    'GUI刷新 单图处理'
    def _update_progress_single_operation(self):
        # 如果不是“繁忙状态”，你也可以做自定义的滚动条效果
        pass  # 当前设置了 0,0，不需要手动更新 value
    def _update_time_label_single_operation(self):
        self._operation_seconds += 0.1              # 每次增加0.1秒

        total_seconds   = self._operation_seconds   # 计算分钟、秒和毫秒
        minutes         = int(total_seconds // 60)
        seconds_with_ms = total_seconds % 60        # 包含小数的秒数（如 5.7 秒）

        time_str = f"{minutes:02d}:{seconds_with_ms:05.2f} | MM:SS.mm"  # 例如 "01:05.70"
        self.timer_label_single_operation.setText(time_str)
    def _start_operation_timer(self):
        self._enable_time_count = True
        self.progress_bar_single_operation.setRange(0, 0)
        self._progress_timer_single_operation.start(100)

        self._operation_seconds = 0
        self.timer_label_single_operation.setText("00:00.00 | MM:SS.mm")
        self._timer_single_operation.start(100)

        self._button_locked()
    def _finish_operation_timer(self):
        if not self._timer_single_operation.isActive():
            return

        self.progress_bar_single_operation.setRange(0, 100)
        self._progress_timer_single_operation.stop()
        self._timer_single_operation.stop()

        total_seconds = self._operation_seconds
        minutes = int(total_seconds // 60)
        seconds_with_ms = total_seconds % 60
        time_str = f"{minutes:02d}:{seconds_with_ms:05.2f} | MM:SS.mm"
        self.timer_label_single_operation.setText(time_str)

        msg_time_usage = f"Time usage: {total_seconds:.2f} /s"
        self.statusBar().showMessage(f"[Operation] {msg_time_usage}")
        msg_html = f'<span style="color:#777777;">[Operation] {msg_time_usage}</span>'  # #FF00FF, #00C400
        self.log_output.append(msg_html)

        self._enable_time_count = False
        self._button_released()

    'GUI刷新 多图处理'
    def _handle_multi_processing(self, operation_name, image, status):
        ''''''
        '''开始处理、完全处理完毕'''
        if isinstance(image, np.ndarray):
            self._update_image_log(status, color = '#019858')
            self._button_released()
            return

        '''处理单张'''
        if not self._timer_multi_images.isActive():
            self._start_multi_images()
            self._update_image_log(status, color = '#0000ff')
            self.log_output.append('<span style="color:#777777;">[Operation] </span>') # 记录操作
            return

        self._print_current_operation(status)

        if status == 'Done.':
            self._finish_multi_images()
            self._image_count(image)
    def _update_progress_multi_images(self):
        # 如果不是“繁忙状态”，你也可以做自定义的滚动条效果
        pass  # 当前设置了 0,0，不需要手动更新 value
    def _update_time_label_multi_images(self):
        self._operation_seconds += 0.1              # 每次增加0.1秒

        total_seconds   = self._operation_seconds   # 计算分钟、秒和毫秒
        minutes         = int(total_seconds // 60)
        seconds_with_ms = total_seconds % 60        # 包含小数的秒数（如 5.7 秒）

        time_str = f"{minutes:02d}:{seconds_with_ms:05.2f} | MM:SS.mm"  # 例如 "01:05.70"
        self.timer_label_multi_images.setText(time_str)
    def _start_multi_images(self):
        self._enable_time_count = True

        self.progress_bar_multi_images.setRange(0, 0)
        self._progress_timer_multi_images.start(100)

        self._operation_seconds = 0
        self.timer_label_multi_images.setText("00:00.00 | MM:SS.mm")
        self._timer_multi_images.start(100)
    def _finish_multi_images(self):
        self.progress_bar_multi_images.setRange(0, 100)
        self._progress_timer_multi_images.stop()
        self._timer_multi_images.stop()

        total_seconds   = self._operation_seconds
        minutes         = int(total_seconds // 60)
        seconds_with_ms = total_seconds % 60
        time_str = f"{minutes:02d}:{seconds_with_ms:05.2f} | MM:SS.mm"
        self.timer_label_multi_images.setText(time_str)

        msg_time_usage  = f" | Time {total_seconds:.2f} /s"
        self.log_output.insertPlainText(msg_time_usage)

        self._enable_time_count = False
    def _print_current_operation(self, status):
        msg_html = f'<span style="color:#777777;">{status}</span>'
        self.log_output.insertHtml(msg_html)
    def _image_count(self, image):
        images_num = f"{image[0]} / {image[1]}"
        self.images_num_label.setText(images_num)

        pass

    '按钮操作'
    def log_on_button_clicked(self, btn_name, color = 'blue'):
        msg_html = f'<span style="color:{color};">[Operation] {btn_name} ... ...</span>'
        self.log_output.append(msg_html)
        self.statusBar().showMessage(f"[Operation] {btn_name} ... ...")
    def _button_locked(self):
        self.btn_open_image.setEnabled(False)
        self.btn_preprocessing.setEnabled(False)
        self.btn_spectrum.setEnabled(False)
        self.btn_reconstruction.setEnabled(False)
        self.btn_focusing.setEnabled(False)
        self.btn_segmentation.setEnabled(False)
        self.btn_identification.setEnabled(False)
        self.btn_phase.setEnabled(False)
        self.btn_polarization.setEnabled(False)
        self.btn_data_summary.setEnabled(False)
        self.btn_all_in_one.setEnabled(False)
        self.btn_multi_images_start.setEnabled(False)
    def _button_released(self):
        self.btn_open_image.setEnabled(True)
        self.btn_preprocessing.setEnabled(True)
        self.btn_spectrum.setEnabled(True)
        self.btn_reconstruction.setEnabled(True)
        self.btn_focusing.setEnabled(True)
        self.btn_segmentation.setEnabled(True)
        self.btn_identification.setEnabled(True)
        self.btn_phase.setEnabled(True)
        self.btn_polarization.setEnabled(True)
        self.btn_data_summary.setEnabled(True)
        self.btn_all_in_one.setEnabled(True)
        self.btn_multi_images_start.setEnabled(True)

    '保存/导入操作'
    def log_on_save_load(self, operation_name: str, message: str):

        msg_html   = f'<span style="color:#4caf50;">[Operation] {message}</span>'    # #FF00FF, #00C400
        self.log_output.append(msg_html)
        self.statusBar().showMessage(message)

        if operation_name == "Save Config":
            pass
        elif operation_name == "Load Config":
            self._update_gui_param_by_load()
        elif operation_name == "Save Data":
            pass
        elif operation_name == "Load Data":
            pass
        else:
            pass
    def _update_gui_param_by_load(self):
        """
        根据配置更新所有 GUI 控件的值。
        支持 config 的多层嵌套结构，input_fields 的 key 用点连接。
        """
        for key_path, widget in self.input_fields.items():
            keys = key_path.split('.')
            value = self.config
            try:
                # 递归访问 config 对应字段
                for key in keys:
                    if isinstance(value, dict):
                        value = value[key]
                    else:
                        value = getattr(value, key)
            except (KeyError, TypeError):
                print(f"[警告] 找不到配置项: {key_path}")
                continue

            # 设置控件值
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index != -1:
                    widget.setCurrentIndex(index)
                else:
                    print(f"[警告] QComboBox 中找不到对应项: {value}，key_path: {key_path}")
            else:
                print(f"[警告] 不支持的控件类型: {type(widget)}，key_path: {key_path}")

    '日志输出'
    def _log_output_append(self, text: str):
        self.log_output.append(text)

    '''这个没用到'''
    def load_config(self, path="config.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        except Exception as e:
            print(f"Fail to load config file: {e}")
            self.config = {}

def main():
    app = QApplication(sys.argv)
    config = HoloConfig()

    processor = Holo_Processor(config)
    processor_thread = QThread()
    processor.moveToThread(processor_thread)
    processor_thread.start()

    window = MainWindow(config)
    controller = Holo_Controller(window, processor)

    window.resize(1525, 825)
    window.show()

    app.aboutToQuit.connect(processor_thread.quit)
    app.aboutToQuit.connect(processor_thread.wait)

    sys.exit(app.exec())

if __name__ == '__main__':

    main()
