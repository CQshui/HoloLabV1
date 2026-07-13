"""
单独的电机控制
极限脉冲位置为 -2500 ~ 60000
"""
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QTextEdit, QStatusBar, QMessageBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtCore import pyqtSignal, QObject
from pymodbus.client import ModbusSerialClient as ModbusClient
from enum import Enum, IntEnum
import serial.tools.list_ports
import time


class SV113Controller:
    """SV113步进驱动器控制类"""
    class FunctionCodes(IntEnum):
        READ_HOLDING_REGISTERS = 0x03
        WRITE_SINGLE_REGISTER = 0x06
        WRITE_MULTIPLE_REGISTERS = 0x10

    class Registers(Enum):
        # 基本信息寄存器
        HARDWARE_VERSION = 0x0000  # 硬件版本 (DWORD)
        SOFTWARE_VERSION = 0x0002  # 软件版本 (DWORD)
        CURRENT_POSITION = 0x0004  # 电机实时位置 (DWORD, pulses)
        STATUS_REGISTER = 0x0006  # 状态寄存器 (WORD)

        # 通信参数
        SERIAL_TIMEOUT = 0x0008  # 串口超时设置 (WORD, ms)
        BAUDRATE = 0x0009  # 波特率设置 (WORD)

        # 运动控制
        SMOOTHING_CONSTANT = 0x000A  # 平滑常数 (WORD)
        START_SPEED = 0x0096  # 启动速度 (WORD, rpm)
        STOP_SPEED = 0x0097  # 停止速度 (WORD, rpm)
        ACCEL_TIME = 0x0098  # 加速时间 (WORD, ms)
        DECEL_TIME = 0x0099  # 减速时间 (WORD, ms)
        TARGET_SPEED = 0x00D8  # 运行目标速度 (DWORD, 0.01rpm)
        ACTUAL_SPEED = 0x00D6  # 实际运行速度 (DWORD, 0.01rpm)

        # 位置控制
        ABSOLUTE_POSITION = 0x00D0  # 运行到绝对位置 (DWORD, pulses)
        RELATIVE_POSITION = 0x00CE  # 运行相对脉冲数 (DWORD, pulses)
        SET_POSITION = 0x00D2  # 设置当前电机位置 (DWORD, pulses)

        # 限位设置
        NEGATIVE_LIMIT = 0x006E  # 软件负限位 (DWORD, pulses)
        POSITIVE_LIMIT = 0x0070  # 软件正限位 (DWORD, pulses)

        # 控制命令
        RUN_STOP = 0x00C8  # 运行/停止 (WORD)
        HOMING = 0x00C9  # 回原点执行 (WORD)
        JOG = 0x00CA  # 点动控制 (WORD)
        SAVE_SETTINGS = 0x00DC  # 断电保存命令 (WORD)

        # 电机参数
        RATED_CURRENT = 0x000D  # 额定电流 (WORD, 0.01A)
        MICROSTEPS = 0x0024  # 细分设置 (DWORD, pulses/rev)

    class Commands(Enum):
        STOP = 0x0000  # 减速停止
        START_FORWARD = 0x0001  # 正向运行
        START_REVERSE = 0x0101  # 反向运行
        EMERGENCY_STOP = 0x0100  # 急停

    class StatusBits(Enum):
        INPUT_0 = 0
        INPUT_1 = 1
        INPUT_2 = 2
        INPUT_3 = 3
        INPUT_4 = 4
        INPUT_5 = 5
        INPUT_6 = 6
        INPUT_7 = 7
        RUN_STATUS = 8  # 位8-9: 00=空闲, 01=启动中, 10=停止中, 11=运行中
        POSITION_WARNING = 10  # 位置超差警告
        IN_POSITION = 12  # 到位输出标识
        NEG_LIMIT = 13  # 负限位
        POS_LIMIT = 14  # 正限位
        HOME_COMPLETE = 15  # 原点完成标志

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
        self.pulses_per_rev = 4000  # 默认细分：4000脉冲/转
        self.lead_mm = 4            # 丝杠导程：5mm/转

        if not self.client.connect():
            raise ConnectionError(f"无法连接到端口 {port}")

    def set_mechanical_params(self, pulses_per_rev, lead_mm):
        """设置机械参数"""
        self.pulses_per_rev = pulses_per_rev
        self.lead_mm = lead_mm

    def pulses_to_mm(self, pulses):
        """脉冲数转毫米"""
        return (pulses / self.pulses_per_rev) * self.lead_mm

    def mm_to_pulses(self, mm):
        """毫米转脉冲数"""
        return int((mm / self.lead_mm) * self.pulses_per_rev)

    def wait_for_position(self, timeout=10.0, check_interval=0.1):
        """
        等待电机到达目标位置
        :param timeout: 超时时间(秒)
        :param check_interval: 检查间隔(秒)
        :return: True=到达, False=超时
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            # 检查到位状态位 (位12)
            if (status >> self.StatusBits.IN_POSITION.value) & 0x01:
                return True
            time.sleep(check_interval)
        return False

    def log_message(self, message):
        """简单的日志记录方法"""
        print(f"[SV113Controller] {message}")  # 简单打印到控制台

    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()

    # def check_limits(self):
    #     """检查限位设置"""
    #     try:
    #         neg_limit = self._read_dword(0x006E)
    #         pos_limit = self._read_dword(0x0070)
    #         status = self.get_status()
    #
    #         self.log_message(f"限位设置: 负限位={neg_limit}, 正限位={pos_limit}")
    #         self.log_message(f"限位状态: 负限位={status >> 13 & 1}, 正限位={status >> 14 & 1}")
    #
    #         if (status >> 13) & 1:
    #             raise RuntimeError("负限位已触发")
    #         if (status >> 14) & 1:
    #             raise RuntimeError("正限位已触发")
    #
    #     except Exception as e:
    #         self.log_message(f"限位检查错误: {str(e)}")
    #         raise

    def move_to_position(self, position, speed=30):
        """移动到绝对位置 - 添加范围检查"""
        # 添加位置范围检查
        if position < -50000 or position > 60000:
            self.log_message(f"错误: 目标位置 {position} 超出范围 (-50000 到 60000)")
            return False

        try:
            # 设置目标速度 (0.01rpm单位)
            speed_units = int(speed * 100)  # 转换为0.01rpm单位
            self._write_dword(self.Registers.TARGET_SPEED.value, speed_units)

            # 设置平滑常数（默认250）
            self._write_word(self.Registers.SMOOTHING_CONSTANT.value, 250)

            # 发送绝对位置移动命令（使用0x00E8-0x00E9寄存器）
            self._write_dword(0x00E8, position)

            self.log_message(f"绝对位置移动指令已发送: 位置={position}, 速度={speed}rpm")
            return True

        except Exception as e:
            self.log_message(f"绝对位置移动失败: {str(e)}")
            return False

    def set_position_mode(self):
        """设置为位置控制模式"""
        # 设置运行模式为I/O控制模式 (值=3)
        self._write_word(0x009F, 0x0003)

        # 设置细分模式
        self.set_microsteps(4000)  # 默认4000脉冲/转

        # 禁用点动模式
        self._write_word(self.Registers.JOG.value, 0x0000)

    def move_relative(self, pulses, speed=300):
        """
        相对移动指定脉冲数
        :param pulses: 脉冲数 (正数为正向，负数为反向)
        :param speed: 运行速度 (rpm)
        """
        # 设置目标速度 (0.01rpm单位)
        self._write_dword(self.Registers.TARGET_SPEED.value, int(speed * 100))

        # 发送相对移动命令
        self._write_dword(self.Registers.RELATIVE_POSITION.value, pulses)

    def jog(self, direction, speed=50, stop_mode=0):
        """
        点动控制
        :param direction: 0=正向, 1=反向
        :param speed: 点动速度 (rpm)
        :param stop_mode: 0=减速停止, 1=立即停止
        """
        # 构建点动命令 (参见手册第33页)
        cmd = (direction << 15) | ((speed & 0x1FF) << 6) | (stop_mode << 5) | 0x01
        self._write_word(self.Registers.JOG.value, cmd)

    def stop(self, emergency=False):
        """停止电机"""
        if emergency:
            self._write_word(self.Registers.RUN_STOP.value, self.Commands.EMERGENCY_STOP.value)
        else:
            self._write_word(self.Registers.RUN_STOP.value, self.Commands.STOP.value)

    def homing(self):
        """强制电机到最高位置 - 添加位置验证"""
        try:
            # 1. 停止电机
            self.stop(emergency=True)
            time.sleep(0.1)

            # 2. 设置当前位置为60000 (脉冲)
            safe_position = 60000
            self.set_current_position(safe_position)

            # 3. 验证设置是否成功
            current_pos = self._read_dword(self.Registers.SET_POSITION.value)
            print('current', current_pos)
            if current_pos != safe_position:
                self.log_message(f"警告: 设置位置失败 (设置={safe_position}, 实际={current_pos})")

            # 4. 以最高速度(300rpm)正向点动
            max_speed = 300  # rpm
            jog_duration = 3.5  # 秒

            self.jog(direction=0, speed=max_speed)
            self.log_message(f"开始正向点动，速度 {max_speed}rpm，持续 {jog_duration}秒")

            # 5. 等待点动完成
            time.sleep(jog_duration)
            self.stop()
            self.log_message("点动完成，电机已停止")

            # 7. 设置当前位置为实际位置
            self.set_current_position(current_pos)

            return True

        except Exception as e:
            self.log_message(f"强制到最高位置失败: {str(e)}")
            return False

    def wait_for_homing_complete(self, timeout=30.0, check_interval=0.1):
        """
        等待回零完成
        :param timeout: 超时时间(秒)
        :param check_interval: 检查间隔(秒)
        :return: True=回零完成, False=超时
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            # 检查回零完成状态位 (位15)
            if (status >> self.StatusBits.HOME_COMPLETE.value) & 0x01:
                # 回零完成后自动设置当前位置为0
                self.set_current_position(0)
                return True
            time.sleep(check_interval)
        return False

    def set_microsteps(self, microsteps):
        """设置细分 (每转脉冲数)"""
        if microsteps < 200 or microsteps > 1000000:
            raise ValueError("细分值必须在200-1000000之间")
        # 先停止电机
        self.stop()
        self._write_dword(self.Registers.MICROSTEPS.value, microsteps)

    # 新增方法：检查电机是否停止
    def is_motor_stopped(self):
        """检查电机是否停止"""
        status = self.get_status()
        run_status = (status >> 8) & 0x03  # 位8-9: 00=空闲, 01=启动中, 10=停止中, 11=运行中
        return run_status == 0 or run_status == 2  # 空闲或停止中

    def set_limits(self, neg_limit, pos_limit):
        """设置软件限位"""
        self._write_dword(self.Registers.NEGATIVE_LIMIT.value, neg_limit)
        self._write_dword(self.Registers.POSITIVE_LIMIT.value, pos_limit)

    def set_current_position(self, position):
        """直接设置当前位置为指定值（不移动电机）"""
        try:
            # 1. 确保驱动器处于停止状态
            self.stop(emergency=True)
            time.sleep(0.1)

            # 2. 直接写入位置寄存器 (0x00D2)
            self._write_dword(self.Registers.SET_POSITION.value, position)

            # 3. 强制驱动器接受这个位置值
            self._write_word(0x00DC, 0x0001)  # 保存设置命令

            # 4. 验证设置
            set_value = self._read_dword(self.Registers.SET_POSITION.value)

            # 5. 如果验证失败，尝试原始寄存器写入
            if set_value != position:
                self.log_message("标准设置失败，尝试原始寄存器写入")
                self._raw_write_position(position)
                set_value = self._read_dword(self.Registers.SET_POSITION.value)

            # 6. 最终验证
            if set_value != position:
                raise IOError(f"无法设置位置! 期望 {position}, 实际 {set_value}")

            self.log_message(f"成功设置当前位置为 {position}")
            return True

        except Exception as e:
            self.log_message(f"设置当前位置失败: {str(e)}")
            return False

    def _raw_write_position(self, position):
        """原始寄存器写入 - 绕过可能的驱动器限制"""
        try:
            # 分解32位位置值为两个16位值
            lo_val = position & 0xFFFF
            hi_val = (position >> 16) & 0xFFFF

            # 直接写入寄存器
            self._write_word(0x00D2, lo_val)  # 位置低16位
            self._write_word(0x00D3, hi_val)  # 位置高16位

            self.log_message(f"原始寄存器写入: 低16位={lo_val}(0x{lo_val:04X}), 高16位={hi_val}(0x{hi_val:04X})")

        except Exception as e:
            self.log_message(f"原始写入失败: {str(e)}")
            raise

    def save_settings(self):
        """保存当前参数到EEPROM"""
        self._write_word(self.Registers.SAVE_SETTINGS.value, 0x0001)

    def get_position(self):
        """获取当前位置"""
        return self._read_dword(self.Registers.CURRENT_POSITION.value)

    def get_speed(self):
        """获取当前速度 (rpm)"""
        speed = self._read_dword(self.Registers.ACTUAL_SPEED.value)
        return speed / 100.0  # 转换为rpm

    def get_status(self):
        """获取状态寄存器"""
        return self._read_word(self.Registers.STATUS_REGISTER.value)

    def check_status_bit(self, status, bit):
        """检查状态位"""
        return (status >> bit.value) & 0x01

    def _read_word(self, address):
        """读取单个寄存器 (WORD)"""
        result = self.client.read_holding_registers(
            address=address,
            count=1,
            slave=self.slave_id
        )
        if result.isError():
            raise IOError(f"读取寄存器 {hex(address)} 失败: {result}")
        return result.registers[0]

    def _read_dword(self, address):
        """读取双寄存器 (DWORD) - 修复字节序问题"""
        try:
            result = self.client.read_holding_registers(
                address=address,
                count=2,
                slave=self.slave_id
            )
            if result.isError():
                raise IOError(f"读取寄存器 {hex(address)} 失败: {result}")

            registers = result.registers
            self.log_message(f"读取地址 {hex(address)}: 寄存器值={registers}")

            # 修复：低位在前，高位在后 (Little Endian for register pair)
            value = registers[0] | (registers[1] << 16)

            # 处理32位有符号整数
            if value > 0x7FFFFFFF:  # 如果最高位为1（负数）
                value -= 0x100000000

            # 验证范围
            if value < -100000 or value > 100000:
                self.log_message(f"警告: 读取的值 {value} 超出预期范围")

            return value

        except Exception as e:
            self.log_message(f"读取双字寄存器 {hex(address)} 失败: {str(e)}")
            raise

    def _write_word(self, address, value):
        """写入单个寄存器 (WORD)"""
        result = self.client.write_register(
            address=address,
            value=value,
            slave=self.slave_id
        )
        if result.isError():
            raise IOError(f"写入寄存器 {hex(address)} 失败: {result}")

    def _write_dword(self, address, value):
        """写入双寄存器 (DWORD) - 修复字节序问题"""
        # 处理负数
        if value < 0:
            value += 0x100000000

        # 分解32位值：低位在前，高位在后
        lo_word = value & 0xFFFF
        hi_word = (value >> 16) & 0xFFFF

        self.log_message(f"写入地址 {hex(address)}: 值={value}, 低位={lo_word}, 高位={hi_word}")

        result = self.client.write_registers(
            address=address,
            values=[lo_word, hi_word],  # 修复：低位在前
            slave=self.slave_id
        )
        if result.isError():
            raise IOError(f"写入寄存器 {hex(address)} 失败: {result}")

    def get_realtime_position(self):
        """获取电机实时位置（修复版本）"""
        try:
            # 读取位置寄存器 (0x0004)
            pos = self._read_dword(self.Registers.CURRENT_POSITION.value)

            # 不再需要额外的补码转换，因为_read_dword已经处理了

            # 返回位置值（已经经过范围检查）
            return pos

        except Exception as e:
            self.log_message(f"读取位置失败: {str(e)}")
            return 0  # 返回0作为安全值

    def restart_driver(self):
        """重启驱动器"""
        try:
            # 向0x00D4寄存器的BIT15写入1来重启驱动器
            restart_command = 0x0100  # BIT15 = 1, 其他位为0
            self._write_word(0x00D4, restart_command)
            self.log_message("驱动器重启指令已发送")
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
                self.log_message("驱动器重启指令已发送（连接断开属正常现象）")
                return True
            else:
                # 其他未知错误才认为是真正的失败
                print('unexpected error:', e)
                self.log_message(f"驱动器重启失败: {str(e)}")
                return False


class MotorControlApp(QMainWindow):
    """主应用程序窗口"""
    # 定义自定义信号
    status_signal = pyqtSignal(int, int, int)  # 位置, 运行状态, 回零状态

    def __init__(self):
        super().__init__()
        self.controller = None
        self.init_ui()
        self.init_timer()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("步进电机控制器")
        self.setGeometry(100, 100, 800, 600)

        # 主控件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # 连接设置组
        connection_group = QGroupBox("连接设置")
        connection_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        connection_layout.addWidget(QLabel("端口:"))
        connection_layout.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("刷新端口")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        connection_layout.addWidget(self.refresh_btn)

        self.slave_id_spin = QSpinBox()
        self.slave_id_spin.setRange(1, 64)
        self.slave_id_spin.setValue(1)
        connection_layout.addWidget(QLabel("从站ID:"))
        connection_layout.addWidget(self.slave_id_spin)

        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("115200")
        connection_layout.addWidget(QLabel("波特率:"))
        connection_layout.addWidget(self.baudrate_combo)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        # 参数设置组
        config_group = QGroupBox("参数设置")
        config_layout = QHBoxLayout()

        self.microsteps_spin = QSpinBox()
        self.microsteps_spin.setRange(200, 1000000)
        self.microsteps_spin.setValue(4000)
        config_layout.addWidget(QLabel("细分(脉冲/转):"))
        config_layout.addWidget(self.microsteps_spin)

        self.current_spin = QDoubleSpinBox()
        self.current_spin.setRange(0.1, 6.5)
        self.current_spin.setValue(1.0)
        self.current_spin.setSingleStep(0.1)
        config_layout.addWidget(QLabel("额定电流(A):"))
        config_layout.addWidget(self.current_spin)

        self.set_config_btn = QPushButton("设置参数")
        self.set_config_btn.clicked.connect(self.set_configuration)
        config_layout.addWidget(self.set_config_btn)

        self.save_btn = QPushButton("保存设置")
        self.save_btn.clicked.connect(self.save_configuration)
        config_layout.addWidget(self.save_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # 限位设置组
        limit_group = QGroupBox("限位设置")
        limit_layout = QHBoxLayout()

        self.neg_limit_spin = QSpinBox()
        self.neg_limit_spin.setRange(-50000, 0)
        self.neg_limit_spin.setValue(-10000)
        limit_layout.addWidget(QLabel("负限位:"))
        limit_layout.addWidget(self.neg_limit_spin)

        self.pos_limit_spin = QSpinBox()
        self.pos_limit_spin.setRange(0, 60000)
        self.pos_limit_spin.setValue(10000)
        limit_layout.addWidget(QLabel("正限位:"))
        limit_layout.addWidget(self.pos_limit_spin)

        self.set_limits_btn = QPushButton("设置限位")
        self.set_limits_btn.clicked.connect(self.set_limits)
        limit_layout.addWidget(self.set_limits_btn)

        limit_group.setLayout(limit_layout)
        layout.addWidget(limit_group)

        # 位置控制组
        pos_group = QGroupBox("位置控制")
        pos_layout = QVBoxLayout()

        h_layout = QHBoxLayout()
        self.position_spin = QSpinBox()
        self.position_spin.setRange(-50000, 50000)
        h_layout.addWidget(QLabel("目标位置:"))
        h_layout.addWidget(self.position_spin)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 1000)
        self.speed_spin.setValue(30)
        h_layout.addWidget(QLabel("速度(rpm):"))
        h_layout.addWidget(self.speed_spin)
        pos_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.move_abs_btn = QPushButton("绝对移动")
        self.move_abs_btn.clicked.connect(self.move_absolute)
        h_layout.addWidget(self.move_abs_btn)

        self.move_rel_btn = QPushButton("相对移动")
        self.move_rel_btn.clicked.connect(self.move_relative)
        h_layout.addWidget(self.move_rel_btn)

        self.set_pos_btn = QPushButton("设置当前位置")
        self.set_pos_btn.clicked.connect(self.set_current_position)
        h_layout.addWidget(self.set_pos_btn)
        pos_layout.addLayout(h_layout)

        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        # 在参数设置组中添加
        self.accel_time_spin = QSpinBox()
        self.accel_time_spin.setRange(10, 5000)  # 10-5000ms
        self.accel_time_spin.setValue(300)  # 默认300ms
        config_layout.addWidget(QLabel("加速时间(ms):"))
        config_layout.addWidget(self.accel_time_spin)

        self.decel_time_spin = QSpinBox()
        self.decel_time_spin.setRange(10, 5000)
        self.decel_time_spin.setValue(300)
        config_layout.addWidget(QLabel("减速时间(ms):"))
        config_layout.addWidget(self.decel_time_spin)

        # 点动控制组
        jog_group = QGroupBox("点动控制")
        jog_layout = QHBoxLayout()

        self.jog_speed_spin = QSpinBox()
        self.jog_speed_spin.setRange(1, 300)
        self.jog_speed_spin.setValue(50)
        jog_layout.addWidget(QLabel("速度(rpm):"))
        jog_layout.addWidget(self.jog_speed_spin)

        self.jog_fwd_btn = QPushButton("正向点动")
        self.jog_fwd_btn.clicked.connect(lambda: self.jog_motor(0))
        jog_layout.addWidget(self.jog_fwd_btn)

        self.jog_rev_btn = QPushButton("反向点动")
        self.jog_rev_btn.clicked.connect(lambda: self.jog_motor(1))
        jog_layout.addWidget(self.jog_rev_btn)

        self.jog_stop_btn = QPushButton("停止")
        self.jog_stop_btn.clicked.connect(self.stop_motor)
        jog_layout.addWidget(self.jog_stop_btn)

        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)

        # 回零控制组
        home_group = QGroupBox("回零操作")
        home_layout = QHBoxLayout()

        self.home_speed_spin = QSpinBox()
        self.home_speed_spin.setRange(1, 500)
        self.home_speed_spin.setValue(200)
        home_layout.addWidget(QLabel("回零速度(rpm):"))
        home_layout.addWidget(self.home_speed_spin)

        self.home_dir_combo = QComboBox()
        self.home_dir_combo.addItems(["正向", "反向"])
        home_layout.addWidget(QLabel("方向:"))
        home_layout.addWidget(self.home_dir_combo)

        self.home_btn = QPushButton("回零")
        self.home_btn.clicked.connect(self.home_motor)
        home_layout.addWidget(self.home_btn)

        home_group.setLayout(home_layout)
        layout.addWidget(home_group)

        # 状态显示
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪")

        # 禁用控制按钮直到连接
        self.set_control_enabled(False)


    def init_timer(self):
        """优化状态定时器"""
        self.update_timer = QTimer()
        self.update_timer.setTimerType(Qt.TimerType.PreciseTimer)  # 高精度定时器
        self.update_timer.timeout.connect(self.lightweight_status_update)
        self.update_timer.start(200)  # 200ms更新间隔

    def lightweight_status_update(self):
        """轻量级状态更新（避免卡顿）"""
        if not self.controller:
            return

        try:
            # 仅读取关键状态（非阻塞式）
            position = self.controller.get_position()
            status = self.controller.get_status()

            # 更新UI（使用信号槽避免直接操作UI）
            self.status_signal.emit(
                position,
                (status >> 8) & 0x03,  # 运行状态
                (status >> 15) & 0x01  # 回零状态
            )

        except Exception as e:
            print(f"状态更新轻量级错误: {str(e)}")

    def update_status(self):
        """更新状态信息（带错误恢复机制）"""
        if not self.controller:
            return

        try:
            # 1. 读取位置
            position = self.controller.get_realtime_position()

            # 处理读取失败情况
            if position is None:
                self.position_error_count += 1
                if self.position_error_count > 3:  # 连续3次失败才显示错误
                    self.status_bar.showMessage("位置读取失败 - 检查连接")
                return

            # 重置错误计数器
            self.position_error_count = 0
            self.last_valid_position = position

            # 2. 读取其他状态信息
            speed = self.controller.get_speed()
            status = self.controller.get_status()

            # 3. 解析状态位
            run_status = ["空闲", "启动中", "停止中", "运行中"][(status >> 8) & 0x03]
            in_position = (status >> SV113Controller.StatusBits.IN_POSITION.value) & 0x01
            neg_limit = (status >> SV113Controller.StatusBits.NEG_LIMIT.value) & 0x01
            pos_limit = (status >> SV113Controller.StatusBits.POS_LIMIT.value) & 0x01

            # 4. 更新UI显示
            self.status_text.clear()
            self.status_text.append(f"当前位置: {position} 脉冲")
            self.status_text.append(f"当前速度: {speed:.2f} rpm")
            self.status_text.append(f"运行状态: {run_status}")
            self.status_text.append(f"到位状态: {'已到位' if in_position else '未到位'}")
            self.status_text.append(
                f"限位状态: 负限位{'触发' if neg_limit else '正常'}, 正限位{'触发' if pos_limit else '正常'}")

            # 状态栏简要信息
            self.status_bar.showMessage(
                f"位置: {position} | 速度: {speed:.1f}rpm | 状态: {run_status} | 到位: {'是' if in_position else '否'}")

        except Exception as e:
            # 显示最后已知位置和错误信息
            self.status_text.append(f"\n[错误] 状态更新失败: {str(e)}")
            self.status_text.append(f"最后有效位置: {self.last_valid_position}")
            self.status_bar.showMessage(f"状态更新错误: {str(e)}")

    def init_timer(self):
        """初始化状态更新定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)       # todo
        self.update_timer.start(500)  # 每500ms更新一次状态

    def refresh_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device, port.device)

    def toggle_connection(self):
        """连接/断开连接"""
        if self.controller is None:
            self.connect_device()
        else:
            self.disconnect_device()

    def connect_device(self):
        """连接设备"""
        port = self.port_combo.currentText()
        slave_id = self.slave_id_spin.value()
        baudrate = int(self.baudrate_combo.currentText())

        try:
            self.controller = SV113Controller(port, slave_id, baudrate)
            self.connect_btn.setText("断开连接")
            self.status_bar.showMessage(f"已连接到 {port}")
            self.set_control_enabled(True)
            self.log_message(f"连接到端口 {port}, 从站ID {slave_id}, 波特率 {baudrate}")

            # 读取并显示版本信息
            hw_version = self.controller._read_dword(SV113Controller.Registers.HARDWARE_VERSION.value)
            sw_version = self.controller._read_dword(SV113Controller.Registers.SOFTWARE_VERSION.value)
            self.log_message(f"硬件版本: {hw_version}, 软件版本: {sw_version}")

        except Exception as e:
            self.status_bar.showMessage(f"连接失败: {str(e)}")
            self.log_message(f"连接失败: {str(e)}")
            QMessageBox.critical(self, "连接错误", f"无法连接到设备:\n{str(e)}")

    def disconnect_device(self):
        """断开设备连接"""
        if self.controller:
            self.controller.close()
            self.controller = None
            self.connect_btn.setText("连接")
            self.status_bar.showMessage("已断开连接")
            self.set_control_enabled(False)
            self.log_message("已断开连接")

    def set_control_enabled(self, enabled):
        """设置控制按钮启用状态"""
        self.set_config_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.set_limits_btn.setEnabled(enabled)
        self.move_abs_btn.setEnabled(enabled)
        self.move_rel_btn.setEnabled(enabled)
        self.set_pos_btn.setEnabled(enabled)
        self.jog_fwd_btn.setEnabled(enabled)
        self.jog_rev_btn.setEnabled(enabled)
        self.jog_stop_btn.setEnabled(enabled)
        self.home_btn.setEnabled(enabled)

    def set_configuration(self):
        """设置电机参数"""
        if not self.controller:
            return

        try:
            # 设置细分
            microsteps = self.microsteps_spin.value()
            self.controller.set_microsteps(microsteps)
            self.log_message(f"设置细分为 {microsteps} 脉冲/转")

            # 设置额定电流 (转换为0.01A单位)
            current = int(self.current_spin.value() * 100)
            self.controller._write_word(SV113Controller.Registers.RATED_CURRENT.value, current)
            self.log_message(f"设置额定电流为 {self.current_spin.value()}A")

        except Exception as e:
            self.log_message(f"参数设置失败: {str(e)}")
            QMessageBox.warning(self, "设置错误", f"参数设置失败:\n{str(e)}")

    def save_configuration(self):
        """保存参数到EEPROM"""
        if not self.controller:
            return

        try:
            self.controller.save_settings()
            self.log_message("参数已保存到EEPROM")
            QMessageBox.information(self, "保存成功", "参数已保存到驱动器EEPROM")
        except Exception as e:
            self.log_message(f"保存失败: {str(e)}")
            QMessageBox.warning(self, "保存错误", f"保存参数失败:\n{str(e)}")

    def set_limits(self):
        """设置软件限位"""
        if not self.controller:
            return

        try:
            neg_limit = self.neg_limit_spin.value()
            pos_limit = self.pos_limit_spin.value()
            self.controller.set_limits(neg_limit, pos_limit)
            self.log_message(f"设置限位: 负限位={neg_limit}, 正限位={pos_limit}")
        except Exception as e:
            self.log_message(f"限位设置失败: {str(e)}")
            QMessageBox.warning(self, "设置错误", f"限位设置失败:\n{str(e)}")

    def move_absolute(self):
        """绝对位置移动"""
        if not self.controller:
            return

        try:
            position = self.position_spin.value()
            speed = self.speed_spin.value()
            self.log_message(f"开始绝对移动到位置 {position}, 速度 {speed}rpm")
            self.controller.move_to_position(position, speed)
            # 等待移动完成
            if self.controller.wait_for_position():
                self.log_message("移动完成")
            else:
                self.log_message("移动超时")

        except Exception as e:
            self.log_message(f"移动失败: {str(e)}")
            QMessageBox.warning(self, "移动错误", f"绝对移动失败:\n{str(e)}")

    def move_relative(self):
        """相对位置移动"""
        if not self.controller:
            return

        try:
            position = self.position_spin.value()
            speed = self.speed_spin.value()

            self.log_message(f"开始相对移动 {position}脉冲, 速度 {speed}rpm")
            self.controller.move_relative(position, speed)
            # 等待移动完成（可选）
            self.log_message("移动指令已发送，等待完成...")
        except Exception as e:
            self.log_message(f"移动失败: {str(e)}")
            QMessageBox.warning(self, "移动错误", f"相对移动失败:\n{str(e)}")

    def jog_motor(self, direction):
        """点动控制"""
        if not self.controller:
            return

        try:
            speed = self.jog_speed_spin.value()
            self.controller.jog(direction, speed)
            dir_text = "正向" if direction == 0 else "反向"
            self.log_message(f"开始{dir_text}点动, 速度 {speed}rpm")
        except Exception as e:
            self.log_message(f"点动失败: {str(e)}")
            QMessageBox.warning(self, "点动错误", f"点动控制失败:\n{str(e)}")

    def home_motor(self):
        """重启驱动器作为回零操作"""
        if not self.controller:
            return

        try:
            # 直接重启驱动器
            if self.controller.restart_driver():
                self.log_message("驱动器重启指令已发送")
                self.status_bar.showMessage("驱动器重启中...")

                # 3秒后检查状态
                QTimer.singleShot(3000, self.check_restart_completion)
            else:
                QMessageBox.warning(self, "重启错误", "驱动器重启失败")

        except Exception as e:
            self.log_message(f"驱动器重启失败: {str(e)}")
            QMessageBox.warning(self, "重启错误", f"驱动器重启失败:\n{str(e)}")

    def check_restart_completion(self):
        """检查驱动器重启是否完成"""
        if self.controller:
            self.status_bar.showMessage("驱动器重启完成")
            self.update_status()  # 更新状态显示
        else:
            print("重启失败！！")

    def stop_motor(self):
        """停止电机"""
        if not self.controller:
            return

        try:
            self.controller.stop()
            self.log_message("电机已停止")
        except Exception as e:
            self.log_message(f"停止失败: {str(e)}")
            QMessageBox.warning(self, "停止错误", f"停止电机失败:\n{str(e)}")

    def set_current_position(self):
        """设置当前位置为60000"""
        if not self.controller:
            return

        try:
            # 直接设置为60000
            self.controller.set_current_position(60000)
            self.log_message("当前位置已设置为60000")

            # 验证设置
            current_pos = self.controller.get_position()
            self.log_message(f"验证位置: {current_pos}")

            # 更新UI
            self.position_spin.setValue(current_pos)

        except Exception as e:
            self.log_message(f"设置当前位置失败: {str(e)}")
            QMessageBox.warning(self, "设置错误", f"设置当前位置失败:\n{str(e)}")

    def update_status(self):
        """实时更新电机状态信息（位置、速度、状态位）"""
        if not self.controller:
            return  # 未连接电机时直接返回

        try:
            # 1. 读取当前位置（带错误处理）
            position = self.controller.get_realtime_position()
            if position is None:
                self.status_bar.showMessage("位置读取失败 - 检查连接")
                return

            # 2. 读取当前速度（rpm）
            speed = self.controller.get_speed()  # 单位: rpm
            speed = round(speed, 2) if speed is not None else "N/A"

            # 3. 读取状态寄存器
            status = self.controller.get_status()

            # 4. 解析状态位（根据SV113Controller.StatusBits）
            run_status = ["空闲", "启动中", "停止中", "运行中"][(status >> 8) & 0x03]  # 位8-9
            in_position = (status >> SV113Controller.StatusBits.IN_POSITION.value) & 0x01  # 位12
            neg_limit = (status >> SV113Controller.StatusBits.NEG_LIMIT.value) & 0x01  # 位13
            pos_limit = (status >> SV113Controller.StatusBits.POS_LIMIT.value) & 0x01  # 位14

            # 5. 更新状态文本框
            self.status_text.clear()
            self.status_text.append(f"当前位置: {position} 脉冲")
            self.status_text.append(f"当前速度: {speed} rpm")
            self.status_text.append(f"运行状态: {run_status}")
            self.status_text.append(f"到位状态: {'已到位' if in_position else '未到位'}")
            self.status_text.append(
                f"限位状态: 负限位{'触发' if neg_limit else '正常'}, 正限位{'触发' if pos_limit else '正常'}")

            # 6. 更新状态栏（底部简短信息）
            self.status_bar.showMessage(
                f"位置: {position} | 速度: {speed}rpm | 状态: {run_status} | 到位: {'是' if in_position else '否'}"
            )

        except Exception as e:
            self.status_text.append(f"\n[错误] 状态更新失败: {str(e)}")
            self.status_bar.showMessage(f"状态更新错误: {str(e)}")

    def log_message(self, message):
        """记录消息到状态框"""
        self.status_text.append(message)

    def debug_communication(self, enable=True):
        """启用/禁用通信调试"""
        if enable:
            import logging
            logging.basicConfig()
            self.client.debug = True
            self.logger = logging.getLogger("pymodbus")
            self.logger.setLevel(logging.DEBUG)
        else:
            self.client.debug = False

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.disconnect_device()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MotorControlApp()
    window.show()
    sys.exit(app.exec())
