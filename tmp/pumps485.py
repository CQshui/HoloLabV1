import sys
import threading
import queue
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QMessageBox, QTabWidget, QFrame)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt6.QtGui import QFont
from pymodbus.client import ModbusSerialClient as ModbusClient
from pymodbus.exceptions import ModbusException
import time
import serial.tools.list_ports


class PumpSignals(QObject):
    connection_status = pyqtSignal(int, bool)  # 泵索引, 连接状态
    operation_result = pyqtSignal(int, bool, str)  # 泵索引, 成功/失败, 消息
    status_update = pyqtSignal(int, dict)  # 泵索引, 状态字典


class ModbusWorker(threading.Thread):
    def __init__(self, command_queue, signals):
        super().__init__()
        self.command_queue = command_queue
        self.signals = signals
        self.pumps = {}  # 存储每个泵的连接对象: {泵索引: ModbusClient}
        self.running = True
        self.current_status = {}  # 存储每个泵的状态: {泵索引: 状态字典}

    def run(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                try:
                    pump_idx = command['pump_idx']
                    func = command['func']
                    args = command.get('args', ())
                    kwargs = command.get('kwargs', {})

                    if func == 'connect':
                        self._connect(pump_idx, *args, **kwargs)
                    elif func == 'disconnect':
                        self._disconnect(pump_idx)
                    elif func == 'start_pump':
                        self._start_pump(pump_idx, *args, **kwargs)
                    elif func == 'stop_pump':
                        self._stop_pump(pump_idx)
                    elif func == 'get_status':
                        self._get_status(pump_idx)
                finally:
                    self.command_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.signals.operation_result.emit(-1, False, f"操作错误: {str(e)}")

    def _connect(self, pump_idx, port, baudrate, address):
        # 如果该泵已经连接，先断开
        if pump_idx in self.pumps and self.pumps[pump_idx].connected:
            self.pumps[pump_idx].close()

        # 创建新的连接
        pump = ModbusClient(
            port=port,
            baudrate=baudrate,
            timeout=1,
            stopbits=1,
            bytesize=8,
            parity='N'
        )

        connected = pump.connect()
        self.pumps[pump_idx] = pump
        self.signals.connection_status.emit(pump_idx, connected)

        if connected:
            self.signals.operation_result.emit(pump_idx, True, f"泵 {pump_idx + 1} 成功连接到 {port}")
            self._get_status(pump_idx)
        else:
            self.signals.operation_result.emit(pump_idx, False, f"泵 {pump_idx + 1} 连接失败")

    def _disconnect(self, pump_idx):
        if pump_idx in self.pumps and self.pumps[pump_idx].connected:
            self._stop_pump(pump_idx)
            self.pumps[pump_idx].close()
            self.signals.connection_status.emit(pump_idx, False)
            self.signals.operation_result.emit(pump_idx, True, f"泵 {pump_idx + 1} 已断开连接")
        else:
            self.signals.operation_result.emit(pump_idx, False, f"泵 {pump_idx + 1} 未连接")

    def _start_pump(self, pump_idx, direction, speed):
        if pump_idx not in self.pumps or not self.pumps[pump_idx].connected:
            self.signals.operation_result.emit(pump_idx, False, f"泵 {pump_idx + 1} 未连接")
            return

        try:
            # 设置方向
            value = 0x0001 if direction else 0x0000
            if not self._write_register(pump_idx, 0x0001, value):
                raise Exception("设置方向失败")

            # 设置速度
            speed_value = int(speed * 10)
            if not self._write_register(pump_idx, 0x0002, speed_value):
                raise Exception("设置速度失败")

            # 启动泵
            if not self._write_register(pump_idx, 0x0000, 0x0001):
                raise Exception("启动泵失败")

            self.signals.operation_result.emit(
                pump_idx,
                True,
                f"泵 {pump_idx + 1} 已启动 - 方向: {'正转' if direction else '反转'}, 转速: {speed} RPM"
            )
            self._get_status(pump_idx)

        except Exception as e:
            self.signals.operation_result.emit(pump_idx, False, f"泵 {pump_idx + 1} 启动失败: {str(e)}")

    def _stop_pump(self, pump_idx):
        if pump_idx not in self.pumps or not self.pumps[pump_idx].connected:
            self.signals.operation_result.emit(pump_idx, False, f"泵 {pump_idx + 1} 未连接")
            return

        if self._write_register(pump_idx, 0x0000, 0x0000):
            self.signals.operation_result.emit(pump_idx, True, f"泵 {pump_idx + 1} 已停止")
            self._get_status(pump_idx)
        else:
            self.signals.operation_result.emit(pump_idx, False, f"泵 {pump_idx + 1} 停止失败")

    def _get_status(self, pump_idx):
        if pump_idx not in self.pumps or not self.pumps[pump_idx].connected:
            return

        status = {
            'running': False,
            'direction': None,
            'speed': None,
            'max_speed': None
        }

        # 读取运行状态
        result = self._read_register(pump_idx, 0x0000)
        if result is not None and len(result) > 0:
            status['running'] = (result[0] == 0x0001)

        # 读取旋转方向
        result = self._read_register(pump_idx, 0x0001)
        if result is not None and len(result) > 0:
            status['direction'] = result[0]

        # 读取当前转速
        result = self._read_register(pump_idx, 0x0002)
        if result is not None and len(result) > 0:
            status['speed'] = result[0] / 10.0

        # 读取最大转速
        result = self._read_register(pump_idx, 0x0006)
        if result is not None and len(result) > 0:
            status['max_speed'] = result[0] / 10.0

        self.current_status[pump_idx] = status
        self.signals.status_update.emit(pump_idx, status)

    def _read_register(self, pump_idx, reg_address, count=1):
        try:
            response = self.pumps[pump_idx].read_holding_registers(
                address=reg_address,
                count=count,
                slave=1
            )
            if response.isError():
                return None
            return response.registers
        except ModbusException:
            return None

    def _write_register(self, pump_idx, reg_address, value):
        try:
            response = self.pumps[pump_idx].write_register(
                address=reg_address,
                value=value,
                slave=1
            )
            return not response.isError()
        except ModbusException as e:
            print(e)
            return False

    def stop(self):
        self.running = False
        # 断开所有泵的连接
        for idx, pump in self.pumps.items():
            if pump.connected:
                pump.close()
        self.command_queue.put(None)


class SinglePumpControl(QWidget):
    def __init__(self, pump_idx, command_queue, signals):
        super().__init__()
        self.pump_idx = pump_idx
        self.command_queue = command_queue
        self.signals = signals
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 标题
        title = QLabel(f"泵 #{self.pump_idx + 1} 控制面板")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 连接设置组
        connection_group = QGroupBox("连接设置")
        connection_layout = QVBoxLayout()

        self.port_combo = QComboBox()
        self.port_combo.addItems(self.get_available_ports())
        connection_layout.addWidget(QLabel("串口:"))
        connection_layout.addWidget(self.port_combo)

        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("9600")
        connection_layout.addWidget(QLabel("波特率:"))
        connection_layout.addWidget(self.baudrate_combo)

        self.address_spin = QSpinBox()
        self.address_spin.setRange(1, 247)
        self.address_spin.setValue(1)
        connection_layout.addWidget(QLabel("设备地址:"))
        connection_layout.addWidget(self.address_spin)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        # 控制组
        control_group = QGroupBox("泵控制")
        control_layout = QVBoxLayout()

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["正转", "反转"])
        control_layout.addWidget(QLabel("旋转方向:"))
        control_layout.addWidget(self.direction_combo)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0, 600)
        self.speed_spin.setValue(30)
        self.speed_spin.setSingleStep(1)
        control_layout.addWidget(QLabel("转速 (RPM):"))
        control_layout.addWidget(self.speed_spin)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动")
        self.start_btn.clicked.connect(self.start_pump)
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_pump)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        control_layout.addLayout(btn_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 状态显示组
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("状态: 未连接")
        status_layout.addWidget(self.status_label)

        self.direction_label = QLabel("方向: -")
        status_layout.addWidget(self.direction_label)

        self.speed_label = QLabel("当前转速: - RPM")
        status_layout.addWidget(self.speed_label)

        self.max_speed_label = QLabel("最大转速: - RPM")
        status_layout.addWidget(self.max_speed_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 状态更新定时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.request_status_update)

        # 添加样式
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #4A708B;
                border-radius: 5px;
                margin-top: 0.5em;
                background-color: #F0F8FF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #2E8B57;
            }
            QLabel {
                color: #2F4F4F;
            }
        """)

    def setup_connections(self):
        """设置信号与槽的连接"""
        self.signals.connection_status.connect(self.update_connection_status)
        self.signals.operation_result.connect(self.handle_operation_result)
        self.signals.status_update.connect(self.update_status_display)

    def get_available_ports(self):
        """获取可用的串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports if ports else ["COM1", "COM3", "/dev/ttyUSB0"]

    def toggle_connection(self):
        """切换连接状态"""
        if self.connect_btn.text() == "连接":
            self.connect_pump()
        else:
            self.disconnect_pump()

    def connect_pump(self):
        """连接泵"""
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        address = self.address_spin.value()

        self.command_queue.put({
            'pump_idx': self.pump_idx,
            'func': 'connect',
            'args': (port, baudrate, address)
        })

    def disconnect_pump(self):
        """断开泵连接"""
        self.command_queue.put({
            'pump_idx': self.pump_idx,
            'func': 'disconnect'
        })

    def start_pump(self):
        """启动泵"""
        direction = self.direction_combo.currentText() == "正转"
        speed = self.speed_spin.value()

        self.command_queue.put({
            'pump_idx': self.pump_idx,
            'func': 'start_pump',
            'args': (direction, speed)
        })

    def stop_pump(self):
        """停止泵"""
        self.command_queue.put({
            'pump_idx': self.pump_idx,
            'func': 'stop_pump'
        })

    def request_status_update(self):
        """请求状态更新"""
        if self.connect_btn.text() == "断开连接" and self.command_queue.qsize() < 5:
            self.command_queue.put({
                'pump_idx': self.pump_idx,
                'func': 'get_status'
            })

    def update_connection_status(self, pump_idx, connected):
        """更新连接状态UI"""
        if pump_idx != self.pump_idx:
            return

        if connected:
            self.connect_btn.setText("断开连接")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.status_timer.start(1000)  # 开始状态更新定时器
        else:
            self.connect_btn.setText("连接")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("状态: 未连接")
            self.direction_label.setText("方向: -")
            self.speed_label.setText("当前转速: - RPM")
            self.max_speed_label.setText("最大转速: - RPM")
            self.status_timer.stop()  # 停止状态更新定时器

    def handle_operation_result(self, pump_idx, success, message):
        """处理操作结果"""
        if pump_idx != self.pump_idx:
            return

        if not success:
            QMessageBox.warning(self, f"泵 {pump_idx + 1} 操作失败", message)

    def update_status_display(self, pump_idx, status):
        """更新状态显示"""
        if pump_idx != self.pump_idx:
            return

        running = "运行中" if status['running'] else "停止"
        self.status_label.setText(f"状态: {running}")

        direction = "正转" if status['direction'] == 1 else "反转"
        self.direction_label.setText(f"方向: {direction}")

        self.speed_label.setText(f"当前转速: {status['speed'] or '-'} RPM")
        self.max_speed_label.setText(f"最大转速: {status['max_speed'] or '-'} RPM")


class PumpControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.command_queue = queue.Queue()
        self.signals = PumpSignals()
        self.worker = ModbusWorker(self.command_queue, self.signals)
        self.init_ui()
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle("多泵蠕动泵控制器")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # 标题
        title = QLabel("多泵蠕动泵控制系统")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2E8B57; margin: 10px 0;")
        layout.addWidget(title)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # 创建两个泵的控制面板
        self.pump_controls = []
        for i in range(2):  # 创建两个泵的控制面板
            pump_control = SinglePumpControl(i, self.command_queue, self.signals)
            self.pump_controls.append(pump_control)
            self.tab_widget.addTab(pump_control, f"泵 #{i + 1}")

        # 日志显示组
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #FAFAFA; border: 1px solid #D3D3D3;")
        log_layout.addWidget(self.log_text)

        # 添加清空日志按钮
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.clear_log)
        clear_btn.setStyleSheet("background-color: #FF6347; color: white; padding: 5px;")
        log_layout.addWidget(clear_btn)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 连接日志信号
        self.signals.operation_result.connect(self.log_operation_result)
        self.signals.connection_status.connect(self.log_connection_status)

        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QTabWidget::pane {
                border: 1px solid #C0C0C0;
                background: white;
            }
            QTabBar::tab {
                background: #E0E0E0;
                border: 1px solid #C0C0C0;
                padding: 8px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            QPushButton {
                background-color: #5F9EA0;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4682B4;
            }
            QPushButton:pressed {
                background-color: #4169E1;
            }
            QPushButton:disabled {
                background-color: #A9A9A9;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 3px;
                border: 1px solid #A9A9A9;
                border-radius: 3px;
            }
        """)

    def log_operation_result(self, pump_idx, success, message):
        """记录操作结果到日志"""
        color = "green" if success else "red"
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"<font color='{color}'>[{timestamp}] 泵 {pump_idx + 1}: {message}</font>")

    def log_connection_status(self, pump_idx, connected):
        """记录连接状态到日志"""
        status = "已连接" if connected else "已断开"
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"<font color='blue'>[{timestamp}] 泵 {pump_idx + 1} {status}</font>")

    def clear_log(self):
        """清空日志"""
        self.log_text.clear()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.worker.stop()
        self.worker.join(timeout=1)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PumpControlApp()
    window.show()
    sys.exit(app.exec())