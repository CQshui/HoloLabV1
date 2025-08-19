from pypylon import pylon, genicam
from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QMutex
import numpy as np
import time
import sys
import os
from datetime import datetime
from threading import Lock
from module.config import HoloConfig

# GUI 部分
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QLineEdit, QGroupBox, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QStatusBar)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, QTimer

class GrabThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._running = False
        self.frame_interval = max(5, int(1000 / camera.config.camera['frame_rate']))

    def run(self):
        self._running = True
        while self._running:
            start_time = time.perf_counter()

            try:
                frame = self.camera.grab_frame()
                if frame is not None:
                    # 确保数据是连续且有效的
                    frame = np.ascontiguousarray(frame)
                    if frame.size > 0:
                        self.frame_ready.emit(frame)
            except Exception as e:
                print(f"采集线程错误: {e}")
                continue

            # 精确控制帧率
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed < self.frame_interval:
                self.msleep(int(self.frame_interval - elapsed))

    def stop(self):
        self._running = False
        self.wait(500)  # 等待线程结束
class RecordingThread(QThread):
    finished    = pyqtSignal()
    progress    = pyqtSignal(int, int)  # (当前计数, 总数)
    error       = pyqtSignal(str)

    def __init__(self, camera, record_mode):
        super().__init__()
        self.camera         = camera
        self.record_mode    = record_mode
        self._stop_flag     = False
        self.was_grabbing   = False  # 记录原始状态

    def run(self):
        try:
            # 暂停实时浏览并开始录制
            # Basler相机API不允许同时开启实时浏览、保存图像，或者说多线程同时操作，所以这里必须先停止再开启
            self._prepare_recording()

            if self.record_mode == 'single':
                self._record_single()
            else:
                self._record_batch()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            # 恢复原始状态
            self.finished.emit()
            self._restore_state()

    def stop(self):
        self._stop_flag = True

    def _prepare_recording(self):
        # 检查并保存当前抓取状态
        self.was_grabbing = (self.camera.grab_thread and self.camera.grab_thread.isRunning())

        # 停止实时浏览
        if self.was_grabbing:
            self.camera.stop_grabbing()
            QThread.msleep(100)  # 等待资源释放

        # 确保相机开始抓取
        if not self.camera.camera.IsGrabbing():
            self.camera.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def _restore_state(self):
        # 先停止
        self.camera.camera.StopGrabbing()

        # 恢复实时浏览，延迟一下，确保资源完全释放
        if self.was_grabbing:
            QTimer.singleShot(100, self.camera.start_grabbing)

    def _record_single(self):
        grab_result = None
        try:
            save_path = self.camera.config.camera['save_path']
            os.makedirs(save_path, exist_ok=True)

            save_fmt = self.camera.config.camera['save_format'].lower()
            frame_rate = float(self.camera.config.camera['frame_rate'])
            timeout = int(1000 / frame_rate)

            grab_result = self.camera.camera.RetrieveResult(timeout, pylon.TimeoutHandling_Return)
            if grab_result.GrabSucceeded():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = os.path.join(save_path, f"{timestamp}.{save_fmt}")
                self.camera._save_image(grab_result, filename, save_fmt)

        except Exception as e:
            self.error.emit(f"单张拍摄失败: {str(e)}")
        finally:
            if grab_result:
                grab_result.Release()

    def _record_batch(self):
        save_path   = self.camera.config.camera['save_path']
        save_fmt    = self.camera.config.camera['save_format']
        num_to_save = self.camera.config.camera['num_to_save']
        frame_rate  = float(self.camera.config.camera['frame_rate'])
        timeout     = max(50, int(1000 / frame_rate))

        count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        while count < num_to_save and not self._stop_flag:
            grab_result = None
            try:
                grab_result = self.camera.camera.RetrieveResult(timeout, pylon.TimeoutHandling_Return)
                if grab_result.GrabSucceeded():
                    filename = f"{save_path}/{timestamp}_{count + 1:04d}.{save_fmt}"
                    self.camera._save_image(grab_result, filename, save_fmt)
                    count += 1

                    self.progress.emit(count, num_to_save)  # 发射进度信号

            except Exception as e:
                self.error.emit(f"批量拍摄错误(第{count}张): {str(e)}")
            finally:
                if grab_result:
                    grab_result.Release()

            # 小延迟避免CPU占用过高
            QThread.msleep(10)
class Camera(QObject):
    frame_ready         = pyqtSignal(np.ndarray)
    error_occurred      = pyqtSignal(str)
    recording_finished  = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config         = config
        self.camera         = None
        self.image          = pylon.PylonImage()

        self.recording_thread = None
        self.record_status  = {'break': False}

        self.grab_thread    = None

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    def initialize(self):
        try:
            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            if not devices:
                raise RuntimeError("No Basler Found.")

            self.camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
            self.camera.Open()

            self.config.camera['camera_name'] = self.camera.GetDeviceInfo().GetModelName()
            self.config.camera['sensor_size'] = f"{self.camera.Width.Max} x {self.camera.Height.Max}"
            self.config.camera['image_width'] = self.camera.Width.Max
            self.config.camera['image_height'] = self.camera.Height.Max

            self._initialize_check_camera_type()
            self._initialize_apply_config()

            return True

        except Exception as e:
            self.error_occurred.emit(f"Camera initialization failed：{e}")
            return False
    def _initialize_check_camera_type(self):
        camera_name = self.config.camera['camera_name']
        self.config.camera['enable_balance_white'] = False
        self.config.camera['enable_ultrashort_exposure'] = False

        if  camera_name.endswith('POL'):
            self.config.camera['camera_type'] = 'Polar'
        else:
            self.config.camera['camera_type'] = 'Regular'

        if   camera_name.endswith('cLET'):
            self.config.camera['enable_ultrashort_exposure'] = True
        elif camera_name.endswith('mLET'):
            self.config.camera['enable_ultrashort_exposure'] = True
        elif camera_name.endswith('c'):
            pass
        elif camera_name.endswith('m'):
            pass
    def _initialize_apply_config(self):
        c = self.config.camera

        # 曝光
        self.camera.ExposureAuto.SetValue('Off')
        if c.get('enable_ultrashort_exposure', False):
            self.camera.ExposureTimeMode.SetValue('UltraShort')
        self.camera.ExposureTime.SetValue(c['exposure_time'])

        # 增益
        self.camera.GainAuto.SetValue('Off')
        self.camera.Gain.SetValue(c['gain'])

        # 白平衡
        if c.get('enable_balance_white', False):
            self.camera.BalanceWhiteAuto.SetValue('Off')

        # 帧率
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionFrameRate.SetValue(c['frame_rate'])

        # 分辨率
        self.camera.Width.SetValue(c['image_width'])
        self.camera.Height.SetValue(c['image_height'])
        self.camera.CenterX.SetValue(False)
        self.camera.CenterY.SetValue(False)
        self.camera.OffsetX.SetValue(0)
        self.camera.OffsetY.SetValue(0)

    def update_param(self, key, value):
        """支持 runtime 参数修改（由 Controller 调用）"""
        self.config.camera[key] = value
        try:
            if key == "exposure_time":
                self.camera.ExposureTime.SetValue(int(value))
            elif key == "frame_rate":
                self.camera.AcquisitionFrameRate.SetValue(float(value))
            elif key == "image_width":
                self.camera.Width.SetValue(int(value))
            elif key == "image_height":
                self.camera.Height.SetValue(value)
            elif key == "offset_x":
                self.camera.OffsetX.SetValue(value)
            elif key == "offset_y":
                self.camera.OffsetY.SetValue(value)
            elif key == 'center_x':
                self.camera.CenterX.SetValue(value)
            elif key == 'center_y':
                self.camera.CenterY.SetValue(value)
            elif key == "enable_ultrashort_exposure":
                # Standard, UltraShort, Short(set to the min of the current range)
                # https://docs.baslerweb.com/exposure-time
                if value:
                    self.camera.ExposureTimeMode.SetValue('UltraShort')
                    self.camera.ExposureTime.SetValue(10)
                else:
                    self.camera.ExposureTimeMode.SetValue('UltraShort')
                    self.camera.ExposureTime.SetValue(self.camera.ExposureTime.GetMin())
            elif key == "enable_balance_white":
                if value:
                    self.camera.BalanceWhiteAuto.SetValue('On')
                else:
                    self.camera.BalanceWhiteAuto.SetValue('Off')

            # elif key == "gain":
            #     self.camera.Gain.SetValue(value)
            # elif key == "enable_gain_mode":
            #     # https://zh.docs.baslerweb.com/gain-auto
            #     if value:
            #         self.camera.GainAuto.SetValue('Continuous')
            #     else:
            #         self.camera.GainAuto.SetValue('Off')

        except Exception as e:
            self.error_occurred.emit(f"更新相机参数失败：{key} -> {e}")

    def start_grabbing(self):
        if self.camera and not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            if not self.grab_thread:
                self.grab_thread = GrabThread(self)
                self.grab_thread.frame_ready.connect(self._handle_frame)
            self.grab_thread.start()
    def stop_grabbing(self):
        if self.grab_thread:
            self.grab_thread.stop()
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
    def grab_frame(self):
        frame_rate = float(self.config.camera['frame_rate'])
        timeout    = int(1000 / frame_rate) # max(50, int(1000 / frame_rate))  # 最小50ms超时

        if self.camera and self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(timeout, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                try:
                    image = self.converter.Convert(grab_result)
                    frame = image.GetArray()
                    return frame
                finally:
                    grab_result.Release()
        return None
    def _handle_frame(self, frame):
        """处理帧数据并发射信号"""
        if frame is not None and frame.size > 0:
            self.frame_ready.emit(frame)

    def start_recording(self, record_mode):
        if self.recording_thread and self.recording_thread.isRunning():
            return False

        self.recording_thread = RecordingThread(self, record_mode)
        self.recording_thread.finished.connect(self._on_recording_finished)
        self.recording_thread.error.connect(self.error_occurred)
        self.recording_thread.start()
        return True
    def stop_recording(self):
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
            self.recording_thread.quit()
            self.recording_thread.wait(1000)  # Wait up to 1 second
    def _execute_recording(self, record_mode):
        """实际执行录制的方法（在子线程中执行）"""
        try:
            if record_mode == 'single':
                self._record_single()
            else:
                self._record_batch()
        finally:
            self.recording_finished.emit()
    def _on_recording_finished(self):
        self.recording_thread = None
        self.recording_finished.emit()
    def _save_image(self, grab_result, filename, fmt):

        self.image.AttachGrabResultBuffer(grab_result)
        fmt = fmt.lower()
        if   fmt == 'jpg':
            ipo = pylon.ImagePersistenceOptions()
            quality = self.config.camera.get('jpg_quality', 90)
            quality = max(1, min(100, quality))
            ipo.SetQuality(quality)
            self.image.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
        elif fmt == 'png':
            self.image.Save(pylon.ImageFileFormat_Png, filename)
        elif fmt == 'bmp':
            self.image.Save(pylon.ImageFileFormat_Bmp, filename)
        elif fmt in ['tif', 'tiff']:
            self.image.Save(pylon.ImageFileFormat_Tiff, filename)
        else:
            self.image.Save(pylon.ImageFileFormat_Bmp, filename)

    def close(self):
        self.stop_grabbing()
        if self.camera:
            self.camera.Close()


class Window(QMainWindow):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.initUI()

        self.camera.recording_finished.connect(self.on_recording_finished)
        self.camera.frame_ready.connect(self.display_image)
        self.camera.error_occurred.connect(self.show_error)
        self.is_live_view_active = False  # 添加状态标志

        # 添加鼠标跟踪
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.on_image_mouse_move

    def initUI(self):
        self.setWindowTitle("Basler Camera")
        self.setGeometry(100, 100, 1000, 600)
        self.setMinimumSize(1100, 500)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("状态: 未连接")

        # 添加像素值显示标签
        self.pixel_info_label = QLabel("光标位置: - , 像素值: -")
        self.status_bar.addPermanentWidget(self.pixel_info_label)

        # 主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局 - 水平布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(3, 3, 3, 3)
        main_layout.setSpacing(0)  # 设置两侧间距

        # 初始化左侧控制面板
        self._initUI_params_and_buttons()
        self.control_panel.setMinimumWidth(320)  # 左侧最小宽度
        self.control_panel.setMaximumWidth(320)  # 左侧最大宽度
        main_layout.addWidget(self.control_panel)

        # 初始化右侧实时画面
        self._initUI_live_view()
        self.display_panel.setMinimumWidth(700)  # 右侧最小宽度
        main_layout.addWidget(self.display_panel, 1)  # 设置伸缩因子

    def _initUI_params_and_buttons(self):
        """初始化左侧参数和按钮面板"""
        self.control_panel = QWidget()
        self.control_panel.setMaximumWidth(350)
        control_layout = QVBoxLayout(self.control_panel)

        # 参数设置组
        params_group = QGroupBox("相机参数设置")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        params_layout.setVerticalSpacing(10)  # 设置行间距为15像素
        params_layout.setContentsMargins(5, 5, 5, 5)  # 设置布局边距(左,上,右,下)
        control_layout.addWidget(params_group)

        # 相机信息
        row = 0
        self.camera_name_label = QLabel("未连接")
        self.camera_name_label.setStyleSheet("font-weight: bold;")
        params_layout.addWidget(QLabel("相机型号:"), row, 0)
        params_layout.addWidget(self.camera_name_label, row, 1, 1, 2)
        row += 1

        # 拍摄模式
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['单张拍摄', '批量拍摄'])
        self.mode_combo.currentTextChanged.connect(self.update_record_mode)
        params_layout.addWidget(QLabel("拍摄模式:"), row, 0)
        params_layout.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        # 曝光时间
        exposure_layout = QHBoxLayout()
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 10000)  # 根据相机实际范围调整
        self.exposure_spin.setValue(self.camera.config.camera['exposure_time'])
        self.exposure_spin.setSuffix(" μs")
        exposure_layout.addWidget(self.exposure_spin)

        # 添加应用按钮
        self.apply_exposure_btn = QPushButton("应用")
        self.apply_exposure_btn.clicked.connect(self.apply_exposure_time)
        exposure_layout.addWidget(self.apply_exposure_btn)

        params_layout.addWidget(QLabel("曝光时间:"), row, 0)
        params_layout.addLayout(exposure_layout, row, 1, 1, 2)
        row += 1

        # 帧率设置
        fps_layout = QHBoxLayout()
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.01, 1000)  # 根据相机实际范围调整
        self.fps_spin.setValue(self.camera.config.camera['frame_rate'])
        self.fps_spin.setSuffix(" FPS")
        fps_layout.addWidget(self.fps_spin)

        # 添加应用按钮
        self.apply_fps_btn = QPushButton("应用")
        self.apply_fps_btn.clicked.connect(self.apply_frame_rate)
        fps_layout.addWidget(self.apply_fps_btn)

        params_layout.addWidget(QLabel("拍摄帧率:"), row, 0)
        params_layout.addLayout(fps_layout, row, 1, 1, 2)
        row += 1

        # 保存数量
        self.num_save_spin = QSpinBox()
        self.num_save_spin.setRange(1, 9999)
        self.num_save_spin.setValue(self.camera.config.camera['num_to_save'])
        self.num_save_spin.valueChanged.connect(self.update_num_to_save)
        params_layout.addWidget(QLabel("保存数量:"), row, 0)
        params_layout.addWidget(self.num_save_spin, row, 1, 1, 2)
        row += 1

        # 路径选择
        self.path_edit = QLineEdit(self.camera.config.camera.get('save_path', ''))
        self.path_edit.setReadOnly(True)
        path_btn = QPushButton("选择路径")
        path_btn.clicked.connect(self.select_save_path)
        params_layout.addWidget(QLabel("保存路径:"), row, 0)
        params_layout.addWidget(self.path_edit, row, 1)
        params_layout.addWidget(path_btn, row, 2)
        row += 1

        # 添加弹簧使按钮区域靠下
        control_layout.addStretch()

        # 按钮组
        btn_group = QGroupBox("操作控制")
        btn_layout = QVBoxLayout()
        btn_group.setLayout(btn_layout)
        control_layout.addWidget(btn_group)

        self.connect_btn = QPushButton("连接相机")
        self.live_view_btn = QPushButton("实时浏览")
        self.record_btn = QPushButton("开始记录")

        self.connect_btn.setMinimumHeight(32)
        self.live_view_btn.setMinimumHeight(32)
        self.record_btn.setMinimumHeight(32)

        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.live_view_btn)
        btn_layout.addWidget(self.record_btn)

        self.connect_btn.clicked.connect(self.toggle_connection)
        self.live_view_btn.clicked.connect(self.toggle_live_view)
        self.record_btn.clicked.connect(self.toggle_recording)

        # 初始化禁用按钮
        self.live_view_btn.setEnabled(False)
        self.record_btn.setEnabled(False)

        # 初始化禁用应用按钮
        self.apply_exposure_btn.setEnabled(False)
        self.apply_fps_btn.setEnabled(False)

    def _initUI_live_view(self):
        self.display_panel = QWidget()
        layout = QVBoxLayout(self.display_panel)

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label)

        # 添加提示标签
        # self.zoom_label = QLabel("提示: 在图像上移动光标可查看像素值")
        # self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.zoom_label.setStyleSheet("color: gray; font-size: 10pt;")
        # layout.addWidget(self.zoom_label)

    def initTimer(self):
        # 用于实时显示的定时器
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self.update_frame)
    def update_frame(self):
        frame = self.camera.grab_frame()
        if frame is not None:
            self.display_image(frame)

    def apply_exposure_time(self):
        """应用曝光时间设置"""
        value = self.exposure_spin.value()
        self.camera.update_param('exposure_time', value)
        self.status_bar.showMessage(f"曝光时间已设置为: {value} μs", 3000)

    def apply_frame_rate(self):
        """应用帧率设置"""
        value = self.fps_spin.value()
        self.camera.update_param('frame_rate', value)
        self.status_bar.showMessage(f"帧率已设置为: {value} FPS", 3000)

    def toggle_connection(self):
        if not self.camera.camera or not self.camera.camera.IsOpen():
            success = self.camera.initialize()
            if success:
                self.connect_btn.setText("断开相机")
                self.live_view_btn.setEnabled(True)
                self.record_btn.setEnabled(True)
                camera_name = self.camera.config.camera['camera_name']
                self.camera_name_label.setText(f"{camera_name}")
                self.status_bar.showMessage(f"状态: 已连接 - {camera_name}")

                # 连接相机后启用应用按钮
                self.apply_exposure_btn.setEnabled(True)
                self.apply_fps_btn.setEnabled(True)
            else:
                self.camera_name_label.setText("连接失败")
                self.status_bar.showMessage("状态: 连接失败")

        else:
            self.stop_live_view()
            self.camera.close()
            self.connect_btn.setText("连接相机")
            self.live_view_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
            self.camera_name_label.setText("未连接")
            self.status_bar.showMessage("状态: 已断开")

            # 断开连接后禁用应用按钮
            self.apply_exposure_btn.setEnabled(False)
            self.apply_fps_btn.setEnabled(False)

    def toggle_live_view(self):
        if self.is_live_view_active:
            self.stop_live_view()
            self.live_view_btn.setText("实时浏览")
        else:
            self.start_live_view()
            self.live_view_btn.setText("停止浏览")
    def toggle_recording(self):
        if not hasattr(self, 'is_recording') or not self.is_recording:
            # 开始记录
            if not os.path.exists(self.camera.config.camera['save_path']):
                try:
                    os.makedirs(self.camera.config.camera['save_path'], exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"创建保存目录失败: {str(e)}")
                    return

            self.record_btn.setText("停止记录")
            self.is_recording = True
            self.record_btn.setEnabled(False)

            # 禁用实时浏览按钮避免冲突
            self.live_view_btn.setEnabled(False)

            mode = "单张" if self.camera.config.camera['record_mode'] == 'single' else "批量"
            self.statusBar().showMessage(f"状态: {mode}拍摄中 - {self.path_edit.text()}")

            # 开始录制
            if not self.camera.start_recording(self.camera.config.camera['record_mode']):
                QMessageBox.warning(self, "警告", "已有录制在进行中")
                return

            # 在确认开始录制后，连接进度信号
            if hasattr(self.camera.recording_thread, 'progress'):
                try:
                    self.camera.recording_thread.progress.disconnect()  # 先断开已有连接
                except:
                    pass
                self.camera.recording_thread.progress.connect(self.update_progress)

            # 200ms后重新启用按钮
            QTimer.singleShot(200, lambda: self.record_btn.setEnabled(True))
        else:
            # 停止记录
            self.record_btn.setEnabled(False)
            self.camera.stop_recording()

    def start_live_view(self):
        if not self.camera.camera or not self.camera.camera.IsOpen():
            QMessageBox.warning(self, "警告", "请先连接相机")
            return
        self.camera.start_grabbing()
        self.live_view_btn.setText("停止浏览")
        self.is_live_view_active = True
    def stop_live_view(self):
        self.camera.stop_grabbing()
        self.live_view_btn.setText("实时浏览")
        # 清空显示（可选）
        self.image_label.clear()
        self.image_label.setText("实时显示已停止")
        self.is_live_view_active = False

    def on_image_mouse_move(self, event: QMouseEvent):
        """处理图像区域的鼠标移动事件，显示像素值"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            return

        # 获取鼠标在图像标签上的位置
        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        # 获取图像标签的尺寸和显示区域
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap = self.image_label.pixmap()

        if not pixmap:
            return

        # 计算实际图像在标签中的位置（居中显示）
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        x_offset = (label_width - pixmap_width) // 2
        y_offset = (label_height - pixmap_height) // 2

        # 检查鼠标是否在图像区域内
        if (x < x_offset or x >= x_offset + pixmap_width or
                y < y_offset or y >= y_offset + pixmap_height):
            self.pixel_info_label.setText("光标位置: - , 像素值: -")
            return

        # 计算在原始图像中的位置
        scale_x = self.current_frame.shape[1] / pixmap_width
        scale_y = self.current_frame.shape[0] / pixmap_height

        img_x = int((x - x_offset) * scale_x)
        img_y = int((y - y_offset) * scale_y)

        # 确保坐标在图像范围内
        img_x = max(0, min(img_x, self.current_frame.shape[1] - 1))
        img_y = max(0, min(img_y, self.current_frame.shape[0] - 1))

        # 获取像素值
        pixel_value = self.current_frame[img_y, img_x]

        # 格式化显示
        if len(pixel_value) == 1:  # 灰度图像
            pixel_str = f"Gray: {pixel_value[0]}"
        elif len(pixel_value) == 3:  # 彩色图像
            pixel_str = f"B: {pixel_value[0]}, G: {pixel_value[1]}, R: {pixel_value[2]}"
        else:
            pixel_str = str(pixel_value)

        self.pixel_info_label.setText(f"光标位置: ({img_x}, {img_y}), 像素值: {pixel_str}")

    def display_image(self, img):
        try:
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                return

            # 保存当前帧用于像素值显示
            self.current_frame = img.copy()

            # 确保图像是连续内存且数据类型正确
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            h, w = img.shape[:2]

            # 创建QImage
            if len(img.shape) == 2:  # 灰度图
                q_img = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
            elif len(img.shape) == 3:  # 彩色图
                if img.shape[2] == 3:  # BGR
                    q_img = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
                elif img.shape[2] == 4:  # BGRA
                    q_img = QImage(img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

            # 计算以高度为基准的缩放比例
            target_height = self.image_label.height()
            scale_ratio = target_height / h
            target_width = int(w * scale_ratio)

            # 异步显示更新
            QTimer.singleShot(0, lambda: self._update_display(
                QPixmap.fromImage(q_img),
                target_width,
                target_height
            ))

        except Exception as e:
            print(f"显示图像错误: {str(e)}")
    def _update_display(self, pixmap, width, height):
        """在主线程中安全更新显示"""
        try:
            scaled_pixmap = pixmap.scaled(
                width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"更新显示错误: {str(e)}")

    def update_record_mode(self, mode_text):
        self.camera.config.camera['record_mode'] = 'single' if mode_text == '单张拍摄' else 'multiple'
    def update_exposure_time(self, value):
        self.camera.update_param('exposure_time', value)
    def update_frame_rate(self, value):
        self.camera.update_param('frame_rate', value)
    def update_num_to_save(self, value):
        self.camera.config.camera['num_to_save'] = value
    def select_save_path(self):
        initial_path = self.camera.config.camera.get('save_path', '')
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", initial_path)
        if save_dir:
            self.camera.config.camera['save_path'] = save_dir
            self.path_edit.setText(save_dir)

    def on_recording_finished(self):
        self.record_btn.setText("开始记录")
        self.is_recording = False
        self.record_btn.setEnabled(True)
        self.live_view_btn.setEnabled(True)  # 重新启用实时浏览

        save_path = self.camera.config.camera.get('save_path', '')
        if self.camera.config.camera['record_mode'] == 'single':
            self.statusBar().showMessage(f"状态: 单张拍摄完成 - {save_path}", 3000)
        else:
            num_to_save = self.camera.config.camera['num_to_save']
            self.statusBar().showMessage(f"状态: 多图拍摄完成 - {num_to_save}张 - {save_path}", 3000)

    def update_progress(self, current, total):
        self.statusBar().showMessage(f"正在保存 {current} / {total} ... ...")
    def show_error(self, message):
        QMessageBox.critical(self, "相机错误", message)
        self.stop_live_view()  # 出错时自动停止

def main_gui():
    app = QApplication(sys.argv)

    # 初始化相机和配置
    config = HoloConfig()
    camera = Camera(config)

    # 创建并显示窗口
    window = Window(camera)
    window.show()

    sys.exit(app.exec())

def main_simple():
    config  = HoloConfig()
    camera  = Camera(config)
    success = camera.initialize()

    if success:
        print(f"相机初始化成功：{config.camera['camera_name']}")
    else:
        print("相机初始化失败")

    camera.start_grabbing()

    print(f"保存到：{config.camera['save_path']}")
    os.makedirs(config.camera['save_path'], exist_ok=True)

    print("正在抓取单帧图像...")
    camera.start_recording(record_mode='single')
    time.sleep(0.5)

    print("正在批量保存图像...")
    camera.start_recording(record_mode='multiple')

    frame_rate   = config.camera['frame_rate']
    num_to_save  = config.camera['num_to_save']
    time_to_wait = num_to_save / frame_rate
    time.sleep(time_to_wait)

    camera.close()
    print("测试完成")

if __name__ == '__main__':

    # main_simple()
    main_gui()


