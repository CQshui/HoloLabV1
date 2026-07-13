"""
单独的阀控程序
"""
import sys
import serial
import serial.tools.list_ports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QMessageBox)
from PyQt6.QtCore import QTimer
import time

class ModbusController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Modbus设备控制器")
        self.setGeometry(100, 100, 400, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # 串口选择部分
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("选择串口:"))

        self.port_combo = QComboBox()
        port_layout.addWidget(self.port_combo)

        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        port_layout.addWidget(self.btn_refresh)

        self.btn_connect = QPushButton("连接")
        self.btn_connect.clicked.connect(self.toggle_connection)
        port_layout.addWidget(self.btn_connect)

        layout.addLayout(port_layout)

        # 状态显示
        self.status_label = QLabel("状态: 未连接")
        layout.addWidget(self.status_label)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_on = QPushButton("控制设备1")
        self.btn_on.clicked.connect(self.turn_on_1)
        self.btn_on.setEnabled(False)
        btn_layout.addWidget(self.btn_on)

        self.btn_off = QPushButton("关闭设备1")
        self.btn_off.clicked.connect(self.turn_off_1)
        self.btn_off.setEnabled(False)
        btn_layout.addWidget(self.btn_off)

        layout.addLayout(btn_layout)

        central_widget.setLayout(layout)

        # 初始化完成后刷新端口列表
        self.refresh_ports()

    def refresh_ports(self):
        """自动检测可用的串口"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.port_combo.addItem("未检测到串口")
            self.btn_connect.setEnabled(False)
        else:
            for port in ports:
                self.port_combo.addItem(port.device, port.device)
            self.btn_connect.setEnabled(True)

    def toggle_connection(self):
        """连接/断开串口"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_port = None
            self.status_label.setText("状态: 已断开连接")
            self.btn_connect.setText("连接")
            self.btn_on.setEnabled(False)
            self.btn_off.setEnabled(False)
        else:
            selected_port = self.port_combo.currentData()
            if selected_port:
                try:
                    self.serial_port = serial.Serial(
                        selected_port,
                        38400,
                        timeout=1,
                        bytesize=8,
                        parity='N',
                        stopbits=1
                    )
                    self.status_label.setText(f"状态: 已连接 {selected_port}")
                    self.btn_connect.setText("断开")
                    self.btn_on.setEnabled(True)
                    self.btn_off.setEnabled(True)
                except serial.SerialException as e:
                    QMessageBox.critical(self, "连接错误", f"无法连接串口:\n{str(e)}")
                    self.status_label.setText("状态: 连接失败")

    def calculate_crc(self,data: bytes) -> bytes:
        """计算Modbus CRC16并返回低字节在前的结果"""
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

    def build_command(self,device_addr: int, reg_addr: int, value: int) -> bytes:
        """构造Modbus指令（自动计算CRC）"""
        cmd = bytes([
            device_addr,  # 设备地址
            0x06,  # 功能码（写单个寄存器）
            (reg_addr >> 8) & 0xFF,  # 寄存器地址高字节
            reg_addr & 0xFF,  # 寄存器地址低字节
            (value >> 8) & 0xFF,  # 值高字节
            value & 0xFF,  # 值低字节
        ])
        cmd= cmd + self.calculate_crc(cmd)
        return  cmd # 附加CRC

    def turn_on_1(self):
        """同时控制多个继电器（非连续地址）"""
        if not (self.serial_port and self.serial_port.is_open):
            return

        # 定义要控制的继电器列表：设备地址, 寄存器地址, 写入值
        relay_commands = [
            (0x01, 0x000F, 0x0001),
            (0x01, 0x0006, 0x0001),
            (0x01, 0x0005, 0x0001),
        ]

        success_count = 0
        for addr, reg, val in relay_commands:
            try:
                cmd = self.build_command(addr, reg, val)  # 自动生成带CRC的指令
                self.serial_port.write(cmd)
                success_count += 1
                time.sleep(0.1)  # Modbus RTU要求帧间隔≥3.5字符时间
            except Exception as e:
                print(f"设备 {addr} 寄存器 {reg:04X} 控制失败: {e}")

        # 更新状态
        self.status_label.setText(f"状态: 已控制 {success_count}/{len(relay_commands)} 个继电器")
        QTimer.singleShot(2000, lambda: self.status_label.setText(f"状态: 已连接 {self.serial_port.port}"))

    def turn_off_1(self):
        """同时控制多个继电器（非连续地址）"""
        if not (self.serial_port and self.serial_port.is_open):
            return

        # 定义要控制的继电器列表：设备地址, 寄存器地址, 写入值
        relay_commands = [
            (0x01, 0x000F, 0x0000),
            (0x01, 0x0006, 0x0000),
            (0x01, 0x0005, 0x0000),
        ]

        success_count = 0
        for addr, reg, val in relay_commands:
            try:
                cmd = self.build_command(addr, reg, val)  # 自动生成带CRC的指令
                self.serial_port.write(cmd)
                success_count += 1
                time.sleep(0.1)  # Modbus RTU要求帧间隔≥3.5字符时间
            except Exception as e:
                print(f"设备 {addr} 寄存器 {reg:04X} 控制失败: {e}")

        # 更新状态
        self.status_label.setText(f"状态: 已控制 {success_count}/{len(relay_commands)} 个继电器")
        QTimer.singleShot(2000, lambda: self.status_label.setText(f"状态: 已连接 {self.serial_port.port}"))

    def closeEvent(self, event):
        """窗口关闭时自动断开连接"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModbusController()
    window.show()
    sys.exit(app.exec())
