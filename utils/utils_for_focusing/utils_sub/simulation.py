import cv2
import numpy as np
import matplotlib.pyplot as plt

def angular_spectrum_propagate(U1, wavelength, z, dx, dy):
    """
    使用角谱衍射公式计算光场传播
    参数：
        U1: 输入平面复振幅 (Ny, Nx)
        wavelength: 波长
        z: 传播距离
        dx, dy: 输入平面采样间隔（x和y方向）
    返回：
        U2: 输出平面复振幅
    """
    Ny, Nx = U1.shape
    k = 2 * np.pi / wavelength

    # 生成频率坐标网格
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))  # x方向空间频率 (1/m)
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))  # y方向空间频率
    FX, FY = np.meshgrid(fx, fy)

    # 计算传递函数 H
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))
    H[np.sqrt((FX**2 + FY**2)) > 1/wavelength] = 0  # 隐逝波处理（可选）

    # 傅里叶变换及频域处理
    A = np.fft.fft2(U1)
    A = np.fft.fftshift(A)  # 将零频移到中心
    U2_spectrum = A * H
    U2 = np.fft.ifft2(np.fft.ifftshift(U2_spectrum))  # 移回原始频域排列后逆变换

    return U2


# 示例使用
if __name__ == "__main__":
    # 参数设置
    wavelength = 0.5e-6  # 波长 500nm
    N = 512             # 采样点数
    L = 0.01            # 计算区域大小 10mm
    dx = L / N          # 采样间隔
    z = -0.01             # 传播距离 0.1m

    # 生成输入场（示例：方形孔径）
    U1 = np.zeros((N, N), dtype=np.complex64)
    U1[N//4:3*N//4, N//4:3*N//4] = 1.0  # 中心区域透光

    # 执行角谱衍射计算
    U2 = angular_spectrum_propagate(U1, wavelength, z, dx, dx)

    # 可视化输出强度
    plt.imshow(np.abs(U2)**2, cmap='gray')
    plt.title('Fig')
    # plt.show()

    result = np.abs(U2)**2
    # 归一化到[0,1]
    result_normalized = (result - result.min()) / (result.max() - result.min())

    # 映射到[0,255]并转换类型
    result_uint8 = (result_normalized * 255).astype(np.uint8)

    # result = result.astype(np.uint8) * 255
    cv2.imwrite(r'F:\dongjiayao\Pycharm\Holo-Track\img\3\tmp\tmp1\off_1_{}.jpg'.format(z), result_uint8)
    # result.save(r'F:\dongjiayao\Pycharm\Holo-Track\img\3\tmp\tmp1\off_1_{}.jpg'.format(z))
