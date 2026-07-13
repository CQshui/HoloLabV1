"""
V0: 泵+阀
"""
import sys
import threading
import queue
import serial
import serial.tools.list_ports
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QMessageBox, QTabWidget, QFrame,
                             QGridLayout)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt6.QtGui import QFont, QColor
from pymodbus.client import ModbusSerialClient as ModbusClient
from pymodbus.exceptions import ModbusException


class AppSignals(QObject):
    connection_status = pyqtSignal(str, int, bool)
    operation_result = pyqtSignal(str, int, bool, str)
    status_update = pyqtSignal(int, dict)


class ModbusWorker(threading.Thread):
    def __init__(self, command_queue, signals):
        super().__init__()
        self.command_queue = command_queue
        self.signals = signals
        self.pumps = {}
        self.valves = {}
        self.running = True
        self.current_status = {}

    def run(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if command is None:
                    continue
                try:
                    device_type = command['device_type']
                    device_idx = command.get('device_idx', 0)
                    func = command['func']
                    args = command.get('args', ())
                    kwargs = command.get('kwargs', {})

                    if device_type == 'pump':
                        if func == 'connect':
                            self._connect_pump(device_idx, *args, **kwargs)
                        elif func == 'disconnect':
                            self._disconnect_pump(device_idx)
                        elif func == 'start_pump':
                            self._start_pump(device_idx, *args, **kwargs)
                        elif func == 'stop_pump':
                            self._stop_pump(device_idx)
                        elif func == 'get_status':
                            self._get_pump_status(device_idx)

                    elif device_type == 'valve':
                        if func == 'connect':
                            self._connect_valve(device_idx, *args, **kwargs)
                        elif func == 'disconnect':
                            self._disconnect_valve(device_idx)
                        elif func == 'control_valves':
                            self._control_valves(device_idx, *args, **kwargs)

                finally:
                    self.command_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.signals.operation_result.emit("system", -1, False, f"操作错误: {str(e)}")

    def _connect_pump(self, pump_idx, port, baudrate, address):
        if pump_idx in self.pumps and self.pumps[pump_idx].connected:
            self.pumps[pump_idx].close()

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
        self.signals.connection_status.emit("pump", pump_idx, connected)

        if connected:
            self.signals.operation_result.emit("pump", pump_idx, True, f"泵 {pump_idx + 1} 成功连接到 {port}")
            self._get_pump_status(pump_idx)
        else:
            self.signals.operation_result.emit("pump", pump_idx, False, f"泵 {pump_idx + 1} 连接失败")

    def _disconnect_pump(self, pump_idx):
        if pump_idx in self.pumps and self.pumps[pump_idx].connected:
            self._stop_pump(pump_idx)
            self.pumps[pump_idx].close()
            self.signals.connection_status.emit("pump", pump_idx, False)
            self.signals.operation_result.emit("pump", pump_idx, True, f"泵 {pump_idx + 1} 已断开连接")
        else:
            self.signals.operation_result.emit("pump", pump_idx, False, f"泵 {pump_idx + 1} 未连接")

    def _start_pump(self, pump_idx, direction, speed):
        if pump_idx not in self.pumps or not self.pumps[pump_idx].connected:
            self.signals.operation_result.emit("pump", pump_idx, False, f"泵 {pump_idx + 1} 未连接")
            return

        try:
            value = 0x0001 if direction else 0x0000
            if not self._write_register(pump_idx, 0x0001, value):
                raise Exception("设置方向失败")

            speed_value = int(speed * 10)
            if not self._write_register(pump_idx, 0x0002, speed_value):
                raise Exception("设置速度失败")

            if not self._write_register(pump_idx, 0x0000, 0x0001):
                raise Exception("启动泵失败")

            self.signals.operation_result.emit(
                "pump",
                pump_idx,
                True,
                f"泵 {pump_idx + 1} 已启动 - 方向: {'正转' if direction else '反转'}, 转速: {speed} RPM"
            )
            self._get_pump_status(pump_idx)

        except Exception as e:
            self.signals.operation_result.emit("pump", pump_idx, False, f"泵 {pump_idx + 1} 启动失败: {str(e)}")

    def _stop_pump(self, pump_idx):
        if pump_idx not in self.pumps or not self.pumps[pump_idx].connected:
            self.signals.operation_result.emit("pump", pump_idx, False, f"泵 {pump_idx + 1} 未连接")
            return

        if self._write_register(pump_idx, 0x0000, 0x0000):
            self.signals.operation_result.emit("pump", pump_idx, True, f"泵 {pump_idx + 1} 已停止")
            self._get_pump_status(pump_idx)
        else:
            self.signals.operation_result.emit("pump", pump_idx, False, f"泵 {pump_idx + 1} 停止失败")

    def _get_pump_status(self, pump_idx):
        if pump_idx not in self.pumps or not self.pumps[pump_idx].connected:
            return

        status = {
            'running': False,
            'direction': None,
            'speed': None,
            'max_speed': None
        }

        result = self._read_register(pump_idx, 0x0000)
        if result:
            status['running'] = (result[0] == 0x0001)

        result = self._read_register(pump_idx, 0x0001)
        if result:
            status['direction'] = result[0]

        result = self._read_register(pump_idx, 0x0002)
        if result:
            status['speed'] = result[0] / 10.0

        result = self._read_register(pump_idx, 0x0006)
        if result:
            status['max_speed'] = result[0] / 10.0

        self.current_status[pump_idx] = status
        self.signals.status_update.emit(pump_idx, status)

    def _read_register(self, pump_idx, reg_address, count=1):
        try:
            response = self.pumps[pump_idx].read_holding_registers(address=reg_address, count=count, slave=1)
            if response.isError():
                return None
            return response.registers
        except ModbusException:
            return None

    def _write_register(self, pump_idx, reg_address, value):
        try:
            response = self.pumps[pump_idx].write_register(address=reg_address, value=value, slave=1)
            return not response.isError()
        except ModbusException:
            return False

    def _connect_valve(self, valve_idx, port, baudrate):
        if valve_idx in self.valves and self.valves[valve_idx].is_open:
            self.valves[valve_idx].close()

        try:
            valve = serial.Serial(port=port, baudrate=baudrate, timeout=1, bytesize=8, parity='N', stopbits=1)
            self.valves[valve_idx] = valve
            self.signals.connection_status.emit("valve", valve_idx, True)
            self.signals.operation_result.emit("valve", valve_idx, True, f"阀门控制器成功连接到 {port}")
        except serial.SerialException as e:
            self.signals.connection_status.emit("valve", valve_idx, False)
            self.signals.operation_result.emit("valve", valve_idx, False, f"阀门控制器连接失败: {str(e)}")

    def _disconnect_valve(self, valve_idx):
        if valve_idx in self.valves and self.valves[valve_idx].is_open:
            self.valves[valve_idx].close()
            self.signals.connection_status.emit("valve", valve_idx, False)
            self.signals.operation_result.emit("valve", valve_idx, True, f"阀门控制器已断开连接")
        else:
            self.signals.operation_result.emit("valve", valve_idx, False, f"阀门控制器未连接")

    def calculate_crc(self, data: bytes) -> bytes:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc.to_bytes(2, byteorder='little')

    def build_command(self, device_addr: int, reg_addr: int, value: int) -> bytes:
        cmd = bytes([
            device_addr,
            0x06,
            (reg_addr >> 8) & 0xFF,
            reg_addr & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        ])
        return cmd + self.calculate_crc(cmd)

    def _control_valves(self, valve_idx, valve_commands):
        if valve_idx not in self.valves or not self.valves[valve_idx].is_open:
            self.signals.operation_result.emit("valve", valve_idx, False, f"阀门控制器未连接")
            return

        success_count = 0
        for addr, reg, val in valve_commands:
            try:
                cmd = self.build_command(addr, reg, val)
                self.valves[valve_idx].write(cmd)
                success_count += 1
                time.sleep(0.1)
            except Exception as e:
                self.signals.operation_result.emit("valve", valve_idx, False,
                                                   f"设备 {addr} 寄存器 {reg:04X} 控制失败: {e}")

        if success_count == len(valve_commands):
            self.signals.operation_result.emit("valve", valve_idx, True, f"成功控制 {success_count} 个阀门")
        else:
            self.signals.operation_result.emit("valve", valve_idx, False,
                                               f"成功控制 {success_count}/{len(valve_commands)} 个阀门")

    def stop(self):
        self.running = False
        for idx, pump in self.pumps.items():
            if pump.connected:
                pump.close()
        for idx, valve in self.valves.items():
            if valve.is_open:
                valve.close()
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
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel(f"泵 #{self.pump_idx + 1} 控制面板")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        connection_group = QGroupBox("连接设置")
        connection_layout = QVBoxLayout()
        connection_layout.setSpacing(5)

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
        connection_layout.addWidget(self.connect_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        control_group = QGroupBox("泵控制")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)

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

        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(5)

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

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.request_status_update)

    def setup_connections(self):
        self.signals.connection_status.connect(self.update_connection_status)
        self.signals.operation_result.connect(self.handle_operation_result)
        self.signals.status_update.connect(self.update_status_display)

    def get_available_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports if ports else ["COM1", "COM3", "/dev/ttyUSB0"]

    def toggle_connection(self):
        if self.connect_btn.text() == "连接":
            self.connect_pump()
        else:
            self.disconnect_pump()

    def connect_pump(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        address = self.address_spin.value()
        self.command_queue.put({
            'device_type': 'pump',
            'device_idx': self.pump_idx,
            'func': 'connect',
            'args': (port, baudrate, address)
        })

    def disconnect_pump(self):
        self.command_queue.put({
            'device_type': 'pump',
            'device_idx': self.pump_idx,
            'func': 'disconnect'
        })

    def start_pump(self):
        direction = self.direction_combo.currentText() == "正转"
        speed = self.speed_spin.value()
        self.command_queue.put({
            'device_type': 'pump',
            'device_idx': self.pump_idx,
            'func': 'start_pump',
            'args': (direction, speed)
        })

    def stop_pump(self):
        self.command_queue.put({
            'device_type': 'pump',
            'device_idx': self.pump_idx,
            'func': 'stop_pump'
        })

    def request_status_update(self):
        if self.connect_btn.text() == "断开连接" and self.command_queue.qsize() < 5:
            self.command_queue.put({
                'device_type': 'pump',
                'device_idx': self.pump_idx,
                'func': 'get_status'
            })

    def update_connection_status(self, device_type, device_idx, connected):
        if device_type != 'pump' or device_idx != self.pump_idx:
            return
        if connected:
            self.connect_btn.setText("断开连接")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.status_timer.start(1000)
        else:
            self.connect_btn.setText("连接")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("状态: 未连接")
            self.direction_label.setText("方向: -")
            self.speed_label.setText("当前转速: - RPM")
            self.max_speed_label.setText("最大转速: - RPM")
            self.status_timer.stop()

    def handle_operation_result(self, device_type, device_idx, success, message):
        if device_type != 'pump' or device_idx != self.pump_idx:
            return
        if not success:
            QMessageBox.warning(self, f"泵 {device_idx + 1} 操作失败", message)

    def update_status_display(self, pump_idx, status):
        if pump_idx != self.pump_idx:
            return
        running = "运行中" if status['running'] else "停止"
        self.status_label.setText(f"状态: {running}")
        direction = "正转" if status['direction'] == 1 else "反转"
        self.direction_label.setText(f"方向: {direction}")
        self.speed_label.setText(f"当前转速: {status['speed'] or '-'} RPM")
        self.max_speed_label.setText(f"最大转速: {status['max_speed'] or '-'} RPM")


class ValveControlWidget(QWidget):
    def __init__(self, command_queue, signals):
        super().__init__()
        self.command_queue = command_queue
        self.signals = signals
        self.valve_idx = 0
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("阀门控制器")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        connection_group = QGroupBox("连接设置")
        connection_layout = QVBoxLayout()

        self.port_combo = QComboBox()
        self.port_combo.addItems(self.get_available_ports())
        connection_layout.addWidget(QLabel("串口:"))
        connection_layout.addWidget(self.port_combo)

        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("38400")
        connection_layout.addWidget(QLabel("波特率:"))
        connection_layout.addWidget(self.baudrate_combo)

        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        btn_layout.addWidget(self.connect_btn)

        self.refresh_btn = QPushButton("刷新串口")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        btn_layout.addWidget(self.refresh_btn)

        connection_layout.addLayout(btn_layout)
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        status_group = QGroupBox("阀门状态")
        status_layout = QGridLayout()
        status_layout.setHorizontalSpacing(5)  # 减小水平间距
        status_layout.setVerticalSpacing(10)  # 适当设置垂直间距

        self.valve_status = {}
        valve_labels = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

        # 使用更紧凑的布局
        for i, label in enumerate(valve_labels):
            # 创建水平布局容器
            valve_container = QWidget()
            valve_layout = QHBoxLayout(valve_container)
            valve_layout.setContentsMargins(0, 0, 0, 0)  # 移除内边距
            valve_layout.setSpacing(5)  # 设置标签和指示灯之间的间距

            # 添加标签
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(30)  # 固定标签宽度
            valve_layout.addWidget(lbl)

            # 添加指示灯
            indicator = QLabel()
            indicator.setFixedSize(20, 20)
            indicator.setStyleSheet("""
                        background-color: gray;
                        border-radius: 10px;
                        border: 1px solid black;
                    """)
            valve_layout.addWidget(indicator)

            # 将容器添加到网格布局
            status_layout.addWidget(valve_container, i // 4, i % 4)
            self.valve_status[i + 1] = indicator

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        control_group = QGroupBox("阀门控制")
        control_layout = QVBoxLayout()

        single_valve_layout = QHBoxLayout()
        single_valve_layout.addWidget(QLabel("选择阀门:"))
        self.valve_combo = QComboBox()
        self.valve_combo.addItems([f"阀门 {i}" for i in range(1, 9)])
        single_valve_layout.addWidget(self.valve_combo)

        self.valve_state_combo = QComboBox()
        self.valve_state_combo.addItems(["打开", "关闭"])
        single_valve_layout.addWidget(self.valve_state_combo)

        self.control_single_btn = QPushButton("控制单个阀门")
        self.control_single_btn.clicked.connect(self.control_single_valve)
        self.control_single_btn.setEnabled(False)
        single_valve_layout.addWidget(self.control_single_btn)

        control_layout.addLayout(single_valve_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        self.valve_status_label = QLabel("状态: 未连接")
        layout.addWidget(self.valve_status_label)

    def setup_connections(self):
        self.signals.connection_status.connect(self.update_connection_status)
        self.signals.operation_result.connect(self.handle_operation_result)

    def get_available_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports if ports else ["COM1", "COM3", "/dev/ttyUSB0"]

    def refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItems(self.get_available_ports())

    def toggle_connection(self):
        if self.connect_btn.text() == "连接":
            self.connect_valve()
        else:
            self.disconnect_valve()

    def connect_valve(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        self.command_queue.put({
            'device_type': 'valve',
            'device_idx': self.valve_idx,
            'func': 'connect',
            'args': (port, baudrate)
        })

    def disconnect_valve(self):
        self.command_queue.put({
            'device_type': 'valve',
            'device_idx': self.valve_idx,
            'func': 'disconnect'
        })

    def control_single_valve(self):
        valve_num = self.valve_combo.currentIndex() + 1
        state = 1 if self.valve_state_combo.currentText() == "打开" else 0
        valve_registers = {1: 0x000F, 2: 0x0006, 3: 0x0005, 4: 0x0004,
                           5: 0x0003, 6: 0x0002, 7: 0x0001, 8: 0x0000}
        reg_addr = valve_registers.get(valve_num, 0x000F)
        self.command_queue.put({
            'device_type': 'valve',
            'device_idx': self.valve_idx,
            'func': 'control_valves',
            'args': ([(0x01, reg_addr, state)],)
        })
        self.update_valve_status(valve_num, state)

    def update_valve_status(self, valve_num, state):
        color = QColor(0, 255, 0) if state == 1 else QColor(255, 0, 0)
        self.valve_status[valve_num].setStyleSheet(
            f"background-color: {color.name()}; border-radius: 10px; border: 1px solid black;"
        )

    def update_connection_status(self, device_type, device_idx, connected):
        if device_type != 'valve' or device_idx != self.valve_idx:
            return
        if connected:
            self.connect_btn.setText("断开连接")
            self.control_single_btn.setEnabled(True)
            self.valve_status_label.setText("状态: 已连接")
        else:
            self.connect_btn.setText("连接")
            self.control_single_btn.setEnabled(False)
            self.valve_status_label.setText("状态: 未连接")
            for indicator in self.valve_status.values():
                indicator.setStyleSheet("background-color: gray; border-radius: 10px;")

    def handle_operation_result(self, device_type, device_idx, success, message):
        if device_type != 'valve' or device_idx != self.valve_idx:
            return
        if not success:
            QMessageBox.warning(self, "阀门操作失败", message)
        else:
            self.valve_status_label.setText(f"状态: {message}")


class IntegratedControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.command_queue = queue.Queue()
        self.signals = AppSignals()
        self.worker = ModbusWorker(self.command_queue, self.signals)
        self.init_ui()
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle("泵阀集成控制系统")
        self.setGeometry(100, 100, 900, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("泵阀集成控制系统")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2E8B57; margin: 10px 0;")
        layout.addWidget(title)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.pump_controls = []
        for i in range(2):
            pump_control = SinglePumpControl(i, self.command_queue, self.signals)
            self.pump_controls.append(pump_control)
            self.tab_widget.addTab(pump_control, f"泵 #{i + 1}")

        valve_control = ValveControlWidget(self.command_queue, self.signals)
        self.tab_widget.addTab(valve_control, "阀门控制")

        # ✅ 新增：系统工况控制页
        scenario_widget = QWidget()
        scenario_layout = QVBoxLayout()
        scenario_widget.setLayout(scenario_layout)

        self.scenarios = {
            1: {"name": "清洗模式", "description": "V1,V3,V5开,泵1正转50RPM,泵2反转30RPM"},
            2: {"name": "进料模式", "description": "V2,V4,V6开,双泵正转40RPM"},
            3: {"name": "反应模式", "description": "V1,V4,V7开,泵1正转30RPM,泵2反转20RPM"},
            4: {"name": "排放模式", "description": "V3,V6,V8开,双泵反转60RPM"},
            0: {"name": "全部关闭", "description": "所有阀门关闭,所有泵停止"}
        }

        btn_grid = QGridLayout()
        for i, (sid, info) in enumerate(self.scenarios.items()):
            btn = QPushButton(f"工况 {sid}\n({info['name']})")
            btn.setToolTip(info["description"])

            # 定义不同按钮的基础颜色
            base_color = "#CD5C5C" if sid == 0 else "#5F9EA0"

            # 设置详细的样式表
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {base_color};
                    color: white;
                    border: 2px outset {base_color};
                    border-radius: 5px;
                    padding: 5px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #{self.lighten_color(base_color, 20)}, stop:1 {base_color});
                    border: 2px outset #{self.lighten_color(base_color, 10)};
                }}
                QPushButton:pressed {{
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 {base_color}, stop:1 #{self.darken_color(base_color, 10)});
                    border: 2px inset {base_color};
                }}
                QToolTip {{
                    background-color: #FFFFCC;
                    color: black;
                    border: 1px solid black;
                    padding: 2px;
                }}
            """)

            btn.clicked.connect(lambda _, s=sid: self.activate_full_scenario(s))
            btn_grid.addWidget(btn, i // 2, i % 2)

        scenario_layout.addLayout(btn_grid)
        self.tab_widget.addTab(scenario_widget, "系统工况")

        # 系统日志
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            background-color: #FAFAFA;
            border: 1px solid #D3D3D3;
            font-family: Consolas, Courier New;
            min-height: 100px;
        """)
        log_layout.addWidget(self.log_text)
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_btn)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.signals.operation_result.connect(self.log_operation_result)
        self.signals.connection_status.connect(self.log_connection_status)

    def lighten_color(self, hex_color, percent):
        """ 调亮颜色 """
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        lighter = tuple(min(255, int(c + (255 - c) * percent / 100)) for c in rgb)
        return f"{lighter[0]:02X}{lighter[1]:02X}{lighter[2]:02X}"

    def darken_color(self, hex_color, percent):
        """ 调暗颜色 """
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        darker = tuple(max(0, int(c * (100 - percent) / 100)) for c in rgb)
        return f"{darker[0]:02X}{darker[1]:02X}{darker[2]:02X}"

    def activate_full_scenario(self, scenario_id):
        scenarios = {
            1: {
                'valves': [
                    (0x01, 0x000F, 1), (0x01, 0x0006, 0), (0x01, 0x0005, 1),
                    (0x01, 0x0004, 0), (0x01, 0x0003, 1), (0x01, 0x0002, 0),
                    (0x01, 0x0001, 0), (0x01, 0x0000, 0)
                ],
                'pumps': [(0, True, 50), (1, False, 30)]
            },
            2: {
                'valves': [
                    (0x01, 0x000F, 0), (0x01, 0x0006, 1), (0x01, 0x0005, 0),
                    (0x01, 0x0004, 1), (0x01, 0x0003, 0), (0x01, 0x0002, 1),
                    (0x01, 0x0001, 0), (0x01, 0x0000, 0)
                ],
                'pumps': [(0, True, 40), (1, True, 40)]
            },
            3: {
                'valves': [
                    (0x01, 0x000F, 1), (0x01, 0x0006, 0), (0x01, 0x0005, 0),
                    (0x01, 0x0004, 1), (0x01, 0x0003, 0), (0x01, 0x0002, 0),
                    (0x01, 0x0001, 1), (0x01, 0x0000, 0)
                ],
                'pumps': [(0, True, 30), (1, False, 20)]
            },
            4: {
                'valves': [
                    (0x01, 0x000F, 0), (0x01, 0x0006, 0), (0x01, 0x0005, 1),
                    (0x01, 0x0004, 0), (0x01, 0x0003, 0), (0x01, 0x0002, 1),
                    (0x01, 0x0001, 0), (0x01, 0x0000, 1)
                ],
                'pumps': [(0, False, 60), (1, False, 60)]
            },
            0: {
                'valves': [
                    (0x01, 0x000F, 0), (0x01, 0x0006, 0), (0x01, 0x0005, 0),
                    (0x01, 0x0004, 0), (0x01, 0x0003, 0), (0x01, 0x0002, 0),
                    (0x01, 0x0001, 0), (0x01, 0x0000, 0)
                ],
                'pumps': [(0, True, 0), (1, True, 0)]
            }
        }

        if scenario_id not in scenarios:
            return

        scenario = scenarios[scenario_id]

        # 控制阀门
        self.command_queue.put({
            'device_type': 'valve',
            'device_idx': 0,
            'func': 'control_valves',
            'args': (scenario['valves'],)
        })

        # 控制泵
        for pump_idx, direction, speed in scenario['pumps']:
            if speed == 0:
                self.command_queue.put({
                    'device_type': 'pump',
                    'device_idx': pump_idx,
                    'func': 'stop_pump'
                })
            else:
                self.command_queue.put({
                    'device_type': 'pump',
                    'device_idx': pump_idx,
                    'func': 'start_pump',
                    'args': (direction, speed)
                })

        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(
            f"<font color='blue'>[{timestamp}] 已激活工况 {scenario_id}: {self.scenarios[scenario_id]['name']}</font>")

    def log_operation_result(self, device_type, device_idx, success, message):
        color = "green" if success else "red"
        device_name = ""
        if device_type == "pump":
            device_name = f"泵 {device_idx + 1}"
        elif device_type == "valve":
            device_name = "阀门控制器"
        else:
            device_name = "系统"
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"<font color='{color}'>[{timestamp}] {device_name}: {message}</font>")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def log_connection_status(self, device_type, device_idx, connected):
        status = "已连接" if connected else "已断开"
        device_name = ""
        if device_type == "pump":
            device_name = f"泵 {device_idx + 1}"
        elif device_type == "valve":
            device_name = "阀门控制器"
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"<font color='blue'>[{timestamp}] {device_name} {status}</font>")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.join(timeout=1)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegratedControlApp()
    window.show()
    sys.exit(app.exec())