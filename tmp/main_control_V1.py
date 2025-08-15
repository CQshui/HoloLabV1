import sys
import threading
import queue
from enum import Enum

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

class SV113Controller:
    """SV113步进驱动器控制类 - 添加限位功能"""
    class StatusBits(Enum):
        RUN_STATUS = 8
        IN_POSITION = 12
        NEG_LIMIT = 13
        POS_LIMIT = 14
        HOME_COMPLETE = 15

    def __init__(self, port, slave_id=1, baudrate=115200, timeout=0.1):
        self.slave_id = slave_id
        self.client = ModbusClient(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=timeout
        )
        if not self.client.connect():
            raise ConnectionError(f"无法连接到端口 {port}")

    def close(self):
        if self.client:
            self.client.close()

    def get_position(self):
        """获取当前位置"""
        return self._read_dword(0x0004)

    def get_speed(self):
        """获取当前速度"""
        speed = self._read_dword(0x00D6)
        return speed / 100.0

    def get_status(self):
        """获取状态寄存器"""
        return self._read_word(0x0006)

    def set_limits(self, neg_limit, pos_limit):
        """设置软件限位"""
        try:
            self._write_dword(0x006E, neg_limit)  # 软件负限位
            self._write_dword(0x0070, pos_limit)  # 软件正限位
            return True
        except Exception as e:
            print(f"设置限位失败: {e}")
            return False

    def get_limits(self):
        """获取当前软件限位"""
        try:
            neg_limit = self._read_dword(0x006E)  # 软件负限位
            pos_limit = self._read_dword(0x0070)  # 软件正限位
            return neg_limit, pos_limit
        except Exception as e:
            print(f"读取限位失败: {e}")
            return None, None

    def move_to_position(self, position, speed=30):
        """移动到绝对位置"""
        try:
            speed_units = int(speed * 100)
            self._write_dword(0x00D8, speed_units)
            self._write_word(0x000A, 250)
            self._write_dword(0x00E8, position)
            return True
        except Exception as e:
            print(f"移动失败: {e}")
            return False

    def jog(self, direction, speed=50, stop_mode=0):
        """点动控制"""
        cmd = (direction << 15) | ((speed & 0x1FF) << 6) | (stop_mode << 5) | 0x01
        self._write_word(0x00CA, cmd)

    def stop(self, emergency=False):
        """停止电机"""
        if emergency:
            self._write_word(0x00C8, 0x0100)
        else:
            self._write_word(0x00C8, 0x0000)

    def restart_driver(self):
        """重启驱动器（回零操作）"""
        try:
            # 向0x00D4寄存器的BIT15写入1来重启驱动器
            restart_command = 0x0100  # BIT15 = 1, 其他位为0
            self._write_word(0x00D4, restart_command)
            print("驱动器重启指令已发送")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # 检查是否是预期的重启相关错误（连接断开、超时等）
            expected_errors = [
                'no response received',
                'input/output',
                'timeout',
                'connection',
                'retries'
            ]
            # 如果是预期的重启错误，认为重启成功
            if any(err in error_msg for err in expected_errors):
                print("驱动器重启指令已发送（连接断开属正常现象）")
                return True
            else:
                # 其他未知错误才认为是真正的失败
                print('unexpected error:', e)
                print(f"驱动器重启失败: {str(e)}")
                return False

    def set_current_position(self, position):
        """设置当前位置"""
        try:
            self.stop(emergency=True)
            time.sleep(0.1)
            self._write_dword(0x00D2, position)
            self._write_word(0x00DC, 0x0001)
            return True
        except Exception as e:
            print(f"设置位置失败: {e}")
            return False

    def _read_word(self, address):
        """读取单个寄存器"""
        result = self.client.read_holding_registers(address=address, count=1, slave=self.slave_id)
        if result.isError():
            raise IOError(f"读取寄存器失败")
        return result.registers[0]

    def _read_dword(self, address):
        """读取双寄存器"""
        result = self.client.read_holding_registers(address=address, count=2, slave=self.slave_id)
        if result.isError():
            raise IOError(f"读取寄存器失败")
        registers = result.registers
        value = registers[0] | (registers[1] << 16)
        if value > 0x7FFFFFFF:
            value -= 0x100000000
        return value

    def _write_word(self, address, value):
        """写入单个寄存器"""
        result = self.client.write_register(address=address, value=value, slave=self.slave_id)
        if result.isError():
            raise IOError(f"写入寄存器失败")

    def _write_dword(self, address, value):
        """写入双寄存器"""
        if value < 0:
            value += 0x100000000
        lo_word = value & 0xFFFF
        hi_word = (value >> 16) & 0xFFFF
        result = self.client.write_registers(address=address, values=[lo_word, hi_word], slave=self.slave_id)
        if result.isError():
            raise IOError(f"写入寄存器失败")

class ModbusWorker(threading.Thread):
    def __init__(self, command_queue, signals):
        super().__init__()
        self.command_queue = command_queue
        self.signals = signals
        self.pumps = {}
        self.valves = {}
        self.motors = {}
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

                    elif device_type == 'motor':
                        if func == 'connect':
                            self._connect_motor(device_idx, *args, **kwargs)
                        elif func == 'disconnect':
                            self._disconnect_motor(device_idx)
                        elif func == 'jog':
                            self._jog_motor(device_idx, *args, **kwargs)
                        elif func == 'stop':
                            self._stop_motor(device_idx)
                        elif func == 'homing':
                            self._homing_motor(device_idx)
                        elif func == 'set_limits':
                            self._set_motor_limits(device_idx, *args, **kwargs)
                        elif func == 'get_limits':  # 新增：获取限位状态
                            self._get_motor_limits(device_idx)
                        elif func == 'get_status':
                            self.get_motor_status(device_idx)

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
        for idx, motor in self.motors.items():
            motor.close()
        self.command_queue.put(None)

    # 电机控制
    def _connect_motor(self, motor_idx, port, baudrate, slave_id):
        try:
            motor = SV113Controller(port, slave_id, baudrate)
            self.motors[motor_idx] = motor
            self.signals.connection_status.emit("motor", motor_idx, True)
            self.signals.operation_result.emit("motor", motor_idx, True, f"电机 {motor_idx + 1} 已连接")
        except Exception as e:
            self.signals.connection_status.emit("motor", motor_idx, False)
            self.signals.operation_result.emit("motor", motor_idx, False, f"电机连接失败: {str(e)}")

    def _disconnect_motor(self, motor_idx):
        if motor_idx in self.motors:
            self.motors[motor_idx].close()
            del self.motors[motor_idx]
            self.signals.connection_status.emit("motor", motor_idx, False)
            self.signals.operation_result.emit("motor", motor_idx, True, f"电机 {motor_idx + 1} 已断开")

    def _set_motor_limits(self, motor_idx, neg_limit, pos_limit):
        """设置电机软件限位"""
        if motor_idx not in self.motors:
            self.signals.operation_result.emit("motor", motor_idx, False, "电机未连接")
            return
        try:
            if self.motors[motor_idx].set_limits(neg_limit, pos_limit):
                self.signals.operation_result.emit("motor", motor_idx, True,
                                                   f"限位已设置: 负限位={neg_limit}, 正限位={pos_limit}")
            else:
                self.signals.operation_result.emit("motor", motor_idx, False, "限位设置失败")
        except Exception as e:
            self.signals.operation_result.emit("motor", motor_idx, False, f"限位设置失败: {str(e)}")

    def _get_motor_limits(self, motor_idx):
        """获取电机软件限位"""
        if motor_idx not in self.motors:
            return
        try:
            neg_limit, pos_limit = self.motors[motor_idx].get_limits()
            if neg_limit is not None and pos_limit is not None:
                limit_status = {
                    'neg_limit': neg_limit,
                    'pos_limit': pos_limit
                }
                self.signals.status_update.emit(motor_idx + 2000, limit_status)  # 使用2000+作为限位状态标识
        except Exception as e:
            print(f"电机限位状态获取失败: {e}")

    def _jog_motor(self, motor_idx, direction, speed):
        if motor_idx not in self.motors:
            self.signals.operation_result.emit("motor", motor_idx, False, "电机未连接")
            return
        try:
            self.motors[motor_idx].jog(direction, speed)
            dir_text = "正向" if direction == 0 else "反向"
            self.signals.operation_result.emit("motor", motor_idx, True, f"{dir_text}点动启动")
        except Exception as e:
            self.signals.operation_result.emit("motor", motor_idx, False, f"点动失败: {str(e)}")

    def _stop_motor(self, motor_idx):
        if motor_idx not in self.motors:
            self.signals.operation_result.emit("motor", motor_idx, False, "电机未连接")
            return
        try:
            self.motors[motor_idx].stop()
            self.signals.operation_result.emit("motor", motor_idx, True, "电机已停止")
        except Exception as e:
            self.signals.operation_result.emit("motor", motor_idx, False, f"停止失败: {str(e)}")

    def _homing_motor(self, motor_idx):
        if motor_idx not in self.motors:
            self.signals.operation_result.emit("motor", motor_idx, False, "电机未连接")
            return
        try:
            if self.motors[motor_idx].restart_driver():
                self.signals.operation_result.emit("motor", motor_idx, True, "驱动器重启完成（回零）")
            else:
                self.signals.operation_result.emit("motor", motor_idx, False, "驱动器重启失败")
        except Exception as e:
            self.signals.operation_result.emit("motor", motor_idx, False, f"重启失败: {str(e)}")

    def get_motor_status(self, motor_idx):
        if motor_idx not in self.motors:
            return
        try:
            position = self.motors[motor_idx].get_position()
            speed = self.motors[motor_idx].get_speed()
            status = self.motors[motor_idx].get_status()

            motor_status = {
                'position': position,
                'speed': speed,
                'status': status,
                'running': (status >> 8) & 0x03,
                'in_position': (status >> 12) & 0x01,
                'neg_limit': (status >> 13) & 0x01,
                'pos_limit': (status >> 14) & 0x01,
            }
            print(motor_status)
            self.signals.status_update.emit(motor_idx + 1000, motor_status)  # 使用1000+作为电机状态标识

            return position

        except Exception as e:
            print(f"电机状态获取失败: {e}")


class SinglePumpControlWidget(QWidget):
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


# 创建电机控制界面类
class MotorControlWidget(QWidget):
    def __init__(self, command_queue, signals, worker_ref):
        super().__init__()
        self.command_queue = command_queue
        self.signals = signals
        self.motor_idx = 0
        self.init_ui()
        self.setup_connections()
        self.worker_ref = worker_ref  # 添加对worker的引用


    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(10, 10, 10, 10)

        # 标题
        title = QLabel("步进电机控制")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 连接设置组
        connection_group = QGroupBox("连接设置")
        connection_layout = QVBoxLayout()

        # 端口选择
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.addItems(self.get_available_ports())
        port_layout.addWidget(QLabel("串口:"))
        port_layout.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(self.refresh_btn)
        connection_layout.addLayout(port_layout)

        # 通信参数
        comm_layout = QHBoxLayout()
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("115200")
        comm_layout.addWidget(QLabel("波特率:"))
        comm_layout.addWidget(self.baudrate_combo)

        self.slave_id_spin = QSpinBox()
        self.slave_id_spin.setRange(1, 64)
        self.slave_id_spin.setValue(1)
        comm_layout.addWidget(QLabel("从站ID:"))
        comm_layout.addWidget(self.slave_id_spin)
        connection_layout.addLayout(comm_layout)

        # 连接按钮
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        # 限位设置组
        limit_group = QGroupBox("软件限位设置")
        limit_layout = QVBoxLayout()

        # 负限位设置
        neg_limit_layout = QHBoxLayout()
        self.neg_limit_spin = QSpinBox()
        self.neg_limit_spin.setRange(-100000, 50000)
        self.neg_limit_spin.setValue(-10000)
        neg_limit_layout.addWidget(QLabel("负限位(脉冲):"))
        neg_limit_layout.addWidget(self.neg_limit_spin)
        limit_layout.addLayout(neg_limit_layout)

        # 正限位设置
        pos_limit_layout = QHBoxLayout()
        self.pos_limit_spin = QSpinBox()
        self.pos_limit_spin.setRange(-50000, 100000)
        self.pos_limit_spin.setValue(10000)
        pos_limit_layout.addWidget(QLabel("正限位(脉冲):"))
        pos_limit_layout.addWidget(self.pos_limit_spin)
        limit_layout.addLayout(pos_limit_layout)

        # 限位设置按钮
        limit_btn_layout = QHBoxLayout()
        self.set_limits_btn = QPushButton("设置软件限位")
        self.set_limits_btn.clicked.connect(self.set_limits)
        self.set_limits_btn.setEnabled(False)
        self.set_limits_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        limit_btn_layout.addWidget(self.set_limits_btn)

        self.clear_limits_btn = QPushButton("清除限位")
        self.clear_limits_btn.clicked.connect(self.clear_limits)
        self.clear_limits_btn.setEnabled(False)
        limit_btn_layout.addWidget(self.clear_limits_btn)

        limit_layout.addLayout(limit_btn_layout)

        # 限位状态显示
        limit_status_layout = QHBoxLayout()
        self.current_neg_limit_label = QLabel("当前负限位: -")
        self.current_pos_limit_label = QLabel("当前正限位: -")
        limit_status_layout.addWidget(self.current_neg_limit_label)
        limit_status_layout.addWidget(self.current_pos_limit_label)
        limit_layout.addLayout(limit_status_layout)

        limit_group.setLayout(limit_layout)
        layout.addWidget(limit_group)

        # 点动控制组
        jog_group = QGroupBox("点动控制")
        jog_layout = QVBoxLayout()

        # 点动速度
        jog_speed_layout = QHBoxLayout()
        self.jog_speed_spin = QSpinBox()
        self.jog_speed_spin.setRange(1, 300)
        self.jog_speed_spin.setValue(50)
        jog_speed_layout.addWidget(QLabel("点动速度(rpm):"))
        jog_speed_layout.addWidget(self.jog_speed_spin)
        jog_layout.addLayout(jog_speed_layout)

        # 点动按钮
        jog_btn_layout = QHBoxLayout()
        self.jog_fwd_btn = QPushButton("正向点动")
        self.jog_fwd_btn.clicked.connect(lambda: self.jog_motor(0))
        self.jog_fwd_btn.setEnabled(False)
        jog_btn_layout.addWidget(self.jog_fwd_btn)

        self.jog_rev_btn = QPushButton("反向点动")
        self.jog_rev_btn.clicked.connect(lambda: self.jog_motor(1))
        self.jog_rev_btn.setEnabled(False)
        jog_btn_layout.addWidget(self.jog_rev_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_motor)
        self.stop_btn.setEnabled(False)
        jog_btn_layout.addWidget(self.stop_btn)
        jog_layout.addLayout(jog_btn_layout)

        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)

        # 回零控制组
        home_group = QGroupBox("回零操作")
        home_layout = QVBoxLayout()

        home_info = QLabel("注意：回零操作将重启驱动器")
        home_info.setStyleSheet("color: orange; font-weight: bold;")
        home_layout.addWidget(home_info)

        self.home_btn = QPushButton("执行回零（重启驱动器）")
        self.home_btn.clicked.connect(self.home_motor)
        self.home_btn.setEnabled(False)
        self.home_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E55A2B;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        home_layout.addWidget(self.home_btn)

        home_group.setLayout(home_layout)
        layout.addWidget(home_group)

        # 状态显示组
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("状态: 未连接")
        status_layout.addWidget(self.status_label)

        self.position_label = QLabel("当前位置: - ")
        status_layout.addWidget(self.position_label)

        self.speed_label = QLabel("当前速度: - rpm")
        status_layout.addWidget(self.speed_label)

        self.limit_label = QLabel("限位状态: -")
        status_layout.addWidget(self.limit_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 状态更新定时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.request_status_update)

    def setup_connections(self):
        self.signals.connection_status.connect(self.update_connection_status)
        self.signals.operation_result.connect(self.handle_operation_result)
        self.signals.status_update.connect(self.update_status_display)

    def get_available_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports if ports else ["COM1", "COM3", "/dev/ttyUSB0"]

    def refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItems(self.get_available_ports())

    def toggle_connection(self):
        if self.connect_btn.text() == "连接":
            self.connect_motor()
        else:
            self.disconnect_motor()

    def connect_motor(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        slave_id = self.slave_id_spin.value()
        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'connect',
            'args': (port, baudrate, slave_id)
        })

    def disconnect_motor(self):
        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'disconnect'
        })

    def set_limits(self):
        neg_limit = self.neg_limit_spin.value()
        pos_limit = self.pos_limit_spin.value()

        # 验证限位设置的合理性
        if neg_limit >= pos_limit:
            QMessageBox.warning(self, "参数错误", "负限位值必须小于正限位值！")
            return

        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'set_limits',
            'args': (neg_limit, pos_limit)
        })

    def clear_limits(self):
        # 设置一个很大的范围作为清除限位
        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'set_limits',
            'args': (-999999, 999999)
        })

    def jog_motor(self, direction):
        speed = self.jog_speed_spin.value()
        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'jog',
            'args': (direction, speed)
        })

    def stop_motor(self):
        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'stop'
        })

    def home_motor(self):
        self.command_queue.put({
            'device_type': 'motor',
            'device_idx': self.motor_idx,
            'func': 'homing'
        })

    def request_status_update(self):
        if self.connect_btn.text() == "断开连接":
            self.command_queue.put({
                'device_type': 'motor',
                'device_idx': self.motor_idx,
                'func': 'get_status'
            })
            # 同时请求限位状态
            self.command_queue.put({
                'device_type': 'motor',
                'device_idx': self.motor_idx,
                'func': 'get_limits'
            })

    def update_connection_status(self, device_type, device_idx, connected):
        if device_type != 'motor' or device_idx != self.motor_idx:
            return

        if connected:
            self.connect_btn.setText("断开连接")
            self.set_limits_btn.setEnabled(True)
            self.clear_limits_btn.setEnabled(True)
            self.jog_fwd_btn.setEnabled(True)
            self.jog_rev_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.home_btn.setEnabled(True)
            self.status_timer.start(1000)
            self.status_label.setText("状态: 已连接")
        else:
            self.connect_btn.setText("连接")
            self.set_limits_btn.setEnabled(False)
            self.clear_limits_btn.setEnabled(False)
            self.jog_fwd_btn.setEnabled(False)
            self.jog_rev_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.home_btn.setEnabled(False)
            self.status_timer.stop()
            self.status_label.setText("状态: 未连接")
            self.position_label.setText("当前位置: -")
            self.speed_label.setText("当前速度: - rpm")
            self.limit_label.setText("限位状态: -")
            self.current_neg_limit_label.setText("当前负限位: -")
            self.current_pos_limit_label.setText("当前正限位: -")

    def handle_operation_result(self, device_type, device_idx, success, message):
        if device_type != 'motor' or device_idx != self.motor_idx:
            return
        if not success:
            QMessageBox.warning(self, "电机操作失败", message)

    def update_status_display(self, status_idx, status):
        """更新状态显示"""
        if status_idx == self.motor_idx + 1000:  # 电机状态标识
            self.position_label.setText(f"当前位置: {status['position']} 脉冲")
            self.speed_label.setText(f"当前速度: {status['speed']:.1f} rpm")

            run_states = ["空闲", "启动中", "停止中", "运行中"]
            run_status = run_states[status['running']]
            in_pos = "已到位" if status['in_position'] else "未到位"

            neg_limit = "触发" if status['neg_limit'] else "正常"
            pos_limit = "触发" if status['pos_limit'] else "正常"

            self.status_label.setText(f"状态: {run_status} | {in_pos}")
            self.limit_label.setText(f"限位: 负限位{neg_limit} | 正限位{pos_limit}")

            # 更新软件限位显示
            if 'soft_neg_limit' in status and status['soft_neg_limit'] is not None:
                self.current_neg_limit_label.setText(f"当前负限位: {status['soft_neg_limit']} 脉冲")
            if 'soft_pos_limit' in status and status['soft_pos_limit'] is not None:
                self.current_pos_limit_label.setText(f"当前正限位: {status['soft_pos_limit']} 脉冲")

        elif status_idx == self.motor_idx + 2000:  # 单独的限位状态标识
            self.current_neg_limit_label.setText(f"当前负限位: {status['neg_limit']} 脉冲")
            self.current_pos_limit_label.setText(f"当前正限位: {status['pos_limit']} 脉冲")

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
            pump_control = SinglePumpControlWidget(i, self.command_queue, self.signals)
            self.pump_controls.append(pump_control)
            self.tab_widget.addTab(pump_control, f"泵 #{i + 1}")

        valve_control = ValveControlWidget(self.command_queue, self.signals)
        self.tab_widget.addTab(valve_control, "阀门控制")

        motor_control = MotorControlWidget(self.command_queue, self.signals, self.worker)
        self.tab_widget.addTab(motor_control, "电机控制")

        # 系统工况控制页
        scenario_widget = QWidget()
        scenario_layout = QVBoxLayout()
        scenario_widget.setLayout(scenario_layout)

        self.scenarios = {
            1: {"name": "测量模式", "description": "V1,V3开,V5开30s关闭,电机上升，相机拍摄，泵1反转5RPM,泵2正转600RPM"},
            2: {"name": "浆液路清洗", "description": "V2,V3，V5开,电机下降,泵1反转5RPM,泵2正转600RPM，1min后全部关闭"},
            3: {"name": "沉积流道冲洗", "description": "V4开,电机处于下降位，1min后关闭"},

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
                'name': "测量模式",
                'description': "V1,V3开,V5开30s关闭,电机上升，相机拍摄，泵1反转5RPM,泵2正转600RPM",
                'valves': [
                    # 设置阀门模式
                    (0x01, 0x00A5, 0),  # V1: 普通模式
                    (0x01, 0x00A4, 0),
                    (0x01, 0x00A3, 0),  # V3: 普通模式
                    (0x01, 0x00A2, 0),
                    (0x01, 0x00A1, 4),  # V5: 开固定时长模式(模式5)

                    # 控制阀门状态
                    (0x01, 0x000F, 1),  # V1开
                    (0x01, 0x000E, 0),
                    (0x01, 0x000D, 1),  # V3开
                    (0x01, 0x000C, 0),
                    (0x01, 0x000B, 3001)  # V5开30秒(N=30*100+1=3001)
                ],
                'pumps': [(0, False, 5), (1, True, 600)],
                'motor': [
                    {'func': 'homing'},  # 电机回零
                    {'func': 'set_limits', 'args': (50000, 60000)},  # 设置限位
                    {'func': 'jog', 'args': (1, 50)},  # 点动下降直到负限位
                ]
            },

            2: {
                'name': "浆液路清洗",
                'description': "V2,V3，V5开,电机下降,泵1反转5RPM,泵2正转600RPM，1min后全部关闭",
                'valves': [
                    # 设置阀门模式
                    (0x01, 0x00A5, 0),
                    (0x01, 0x00A4, 4),  # V2: 开固定时长模式
                    (0x01, 0x00A3, 4),  # V3: 开固定时长模式
                    (0x01, 0x00A2, 0),
                    (0x01, 0x00A1, 4),  # V5: 开固定时长模式

                    # 控制阀门状态
                    (0x01, 0x000F, 0),
                    (0x01, 0x000E, 6001),  # V2开60秒(N=60*100+1=6001)
                    (0x01, 0x000D, 6001),  # V3开60秒
                    (0x01, 0x000C, 0),
                    (0x01, 0x000B, 6001)  # V5开60秒
                ],
                'pumps': [(0, False, 5), (1, True, 600)],
                'timers': [
                    (60, {'device_type': 'pump', 'device_idx': 0, 'func': 'stop_pump'}),    # todo 这部分定时有点疑问，工况内部还需要定时吗？是否应该切换工况时定时？
                    (60, {'device_type': 'pump', 'device_idx': 1, 'func': 'stop_pump'})
                ],
                'motor': [
                    {'func': 'set_limits', 'args': (30000, 50000)},  # 设置限位
                    {'func': 'jog', 'args': (1, 50)},  # 点动下降直到负限位
                ]
            },

            3: {
                'name': "沉积流道冲洗",
                'description': "V4开,电机处于下降位，1min后关闭",
                'valves': [
                    # 设置阀门模式
                    (0x01, 0x00A5, 0),  # V1: 普通模式
                    (0x01, 0x00A4, 0),
                    (0x01, 0x00A3, 0),  # V3: 普通模式
                    (0x01, 0x00A2, 4),  # V4: 开固定时长模式
                    (0x01, 0x00A1, 0),  # V5: 开普通模式
                    # 控制阀门状态
                    (0x01, 0x000F, 0),  # V1关
                    (0x01, 0x000E, 0),
                    (0x01, 0x000D, 0),  # V3关
                    (0x01, 0x000C, 6001),  # V4开60秒(N=60*100+1=6001)
                    (0x01, 0x000B, 0),  # V5关

                ],
                'pumps': [(0, False, 0), (1, True, 0)],
                'motor': []  # 电机保持当前位置
            },

            0: {
                'valves': [
                    # 设置阀门模式
                    (0x01, 0x00A5, 0),
                    (0x01, 0x00A4, 0),  # V2: 开固定时长模式
                    (0x01, 0x00A3, 0),  # V3: 开固定时长模式
                    (0x01, 0x00A2, 0),
                    (0x01, 0x00A1, 0),  # V5: 开固定时长模式

                    # 控制阀门状态
                    (0x01, 0x000F, 0),
                    (0x01, 0x000E, 0),  # V2开60秒(N=60*100+1=6001)
                    (0x01, 0x000D, 0),  # V3开60秒
                    (0x01, 0x000C, 0),
                    (0x01, 0x000B, 0)  # V5开60秒
                ],
                'pumps': [(0, True, 0), (1, True, 0)],
                'motor': [
                    {'func': 'homing'}  # 电机回零
                ]
            }
        }

        if scenario_id not in scenarios:
            return

        scenario = scenarios[scenario_id]

        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(
            f"<font color='blue'>[{timestamp}] 已激活工况 {scenario_id}: {self.scenarios[scenario_id]['name']}</font>")

        # 以下为调试内容，正式版删除，并将timestamp上方注释内容取消注释
        # 检查阀门控制器是否连接
        self.command_queue.put({
            'device_type': 'valve',
            'device_idx': 0,
            'func': 'control_valves',
            'args': (scenario['valves'],)
        })
        self.log_text.append(f"<font color='blue'>[{timestamp}] 已发送阀门控制命令</font>")

        # 控制泵
        for pump_idx, direction, speed in scenario['pumps']:
            if speed == 0:
                self.command_queue.put({
                    'device_type': 'pump',
                    'device_idx': pump_idx,
                    'func': 'stop_pump'
                })
                self.log_text.append(f"<font color='blue'>[{timestamp}] 已发送停止泵{pump_idx + 1}命令</font>")
            else:
                self.command_queue.put({
                    'device_type': 'pump',
                    'device_idx': pump_idx,
                    'func': 'start_pump',
                    'args': (direction, speed)
                })
                dir_text = "反转" if direction else "正转"
                self.log_text.append(
                    f"<font color='blue'>[{timestamp}] 已启动泵{pump_idx + 1}: {dir_text} {speed}RPM</font>")

        # 控制电机
        # 电机操作 - 重启驱动器
        if scenario_id == 1 or scenario_id == 0:
            self.command_queue.put({
                'device_type': 'motor',
                'device_idx': 0,
                'func': 'homing'
            })

        # 获取电机控制部件
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, MotorControlWidget):
                motor_control = widget
                break
        else:
            motor_control = None

        if motor_control:
            # 等待电机重启完成后执行点动操作
            # 创建并启动等待线程
            def execute_jog(params):
                """
                执行电机操作序列
                :param params: scenario['motor']参数，格式如 [{'func': 'set_limits', 'args': (50000, 60000)}, ...]
                """
                timestamp = time.strftime("%H:%M:%S")
                self.log_text.append(f"<font color='blue'>[{timestamp}] 开始执行电机操作序列</font>")

                # 执行每个电机命令
                for motor_cmd in params:
                    # 跳过已经执行的homing命令
                    if motor_cmd['func'] == 'homing':
                        continue

                    # 发送命令到队列
                    self.command_queue.put({
                        'device_type': 'motor',
                        'device_idx': 0,
                        'func': motor_cmd['func'],
                        'args': motor_cmd.get('args', ())
                    })

                    # 记录日志
                    args_str = ', '.join(map(str, motor_cmd.get('args', ())))
                    self.log_text.append(
                        f"<font color='blue'>[{timestamp}] 发送电机命令: {motor_cmd['func']}({args_str})</font>")

                self.log_text.append(f"<font color='blue'>[{timestamp}] 电机操作序列执行完成</font>")

            def wait_for_position():
                while True:
                    try:
                        # 获取电机状态
                        status = self.worker.motors[0].get_status()
                        position = self.worker.motors[0].get_position()

                        # 检查是否达到目标位置
                        if position >= 60000:
                            execute_jog(scenario['motor'])
                            break

                        # 短暂休眠避免CPU占用过高
                        time.sleep(0.5)

                    except Exception as e:
                        timestamp = time.strftime("%H:%M:%S")
                        self.log_text.append(
                            f"<font color='red'>[{timestamp}] 获取电机状态错误: {str(e)}</font>")
                        break

            if scenario_id == 1:
                # 启动等待线程
                wait_thread = threading.Thread(target=wait_for_position, daemon=True)
                wait_thread.start()

                timestamp = time.strftime("%H:%M:%S")
                self.log_text.append(
                    f"<font color='blue'>[{timestamp}] 等待电机位置达到60000...</font>")
            else:
                execute_jog(scenario['motor'])

    def log_operation_result(self, device_type, device_idx, success, message):
        color = "green" if success else "red"
        device_name = ""
        if device_type == "pump":
            device_name = f"泵 {device_idx + 1}"
        elif device_type == "valve":
            device_name = "阀门控制器"
        elif device_type == "motor":
            device_name = "步进电机"
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