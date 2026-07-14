from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import center_of_mass
import cupy as cp

class Spectrum:
    def __init__(self, hologram=None, config=None):
        # 输入数据
        self.hologram_raw = hologram.hologram_raw
        self.hologram     = hologram.hologram
        self.spectrum_raw = hologram.spectrum_raw
        self.spectrum     = hologram.spectrum

        self.image_name   = config.file_info['image_name']
        self.holo_type    = config.file_info['holo_type']
        self.method       = config.spectrum['method']
        self.pixel_size   = config.image_info['pixel_size']
        self.wavelength   = config.image_info['wavelength']

        self.roi_center_x = config.spectrum['ROI_rectangle']['center_x']
        self.roi_center_y = config.spectrum['ROI_rectangle']['center_y']
        self.roi_width    = config.spectrum['ROI_rectangle']['rect_width']
        self.roi_height   = config.spectrum['ROI_rectangle']['rect_height']

        # 保留原始对象用于修改
        self._hologram = hologram
        self._config   = config

    def run(self):
        self.spectrum_raw = self.Get_Spectrum()

        '''Inline / Off-Axis'''
        if self.holo_type == 'Inline':
            # Inline 类型直接使用 spectrum_raw
            self.spectrum = self.spectrum_raw
        else:
            if self.method == 'Manual_Select':
                self.spectrum = self.Get_Spectrum_Manual_Select()
                self.move_ROI_to_center()
            elif self.method == 'Give_Values':
                self.spectrum = self.Get_Spectrum_Give_Values()
                self.move_ROI_to_center()
            elif self.method == 'Auto_Define':
                self.spectrum = self.Get_Spectrum_Auto_Define()
                self.move_ROI_to_center()
            else:
                raise ValueError(f"Unknown spectrum method: {self.method}")

        self.modify_hologram_and_config()

    def modify_hologram_and_config(self):
        # 更新 hologram 对象
        self._hologram.spectrum_raw = self.spectrum_raw
        self._hologram.spectrum     = self.spectrum
        self._hologram.status_msg   = 'Spectrum Analysis Done'

        # 更新 config 中 ROI 区域参数
        self._config.spectrum['ROI_rectangle']['center_x']      = self.roi_center_x
        self._config.spectrum['ROI_rectangle']['center_y']      = self.roi_center_y
        self._config.spectrum['ROI_rectangle']['rect_width']    = self.roi_width
        self._config.spectrum['ROI_rectangle']['rect_height']   = self.roi_height

    def move_ROI_to_center(self):
        h, w = self.spectrum.shape

        # 原 ROI 坐标
        x0 = int(self.roi_center_x - self.roi_width // 2)
        y0 = int(self.roi_center_y - self.roi_height // 2)
        x1 = x0 + self.roi_width
        y1 = y0 + self.roi_height

        # 截取 ROI 区域
        roi = self.spectrum[y0:y1, x0:x1]

        # 创建空图并将 ROI 移动到中心
        new_spectrum = np.zeros_like(self.spectrum, dtype=self.spectrum.dtype)
        cx = w // 2
        cy = h // 2

        # 目标区域坐标
        new_x0 = cx - self.roi_width // 2
        new_y0 = cy - self.roi_height // 2
        new_x1 = new_x0 + self.roi_width
        new_y1 = new_y0 + self.roi_height

        new_spectrum[new_y0:new_y1, new_x0:new_x1] = roi

        self.spectrum = new_spectrum

    def Get_Spectrum(self):
        return np.fft.fftshift(np.fft.fft2(self.hologram))

    def Get_Spectrum_Manual_Select(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.canvas.manager.set_window_title('Spectrum ROI Selection Tool')

        ax.imshow(np.log1p(np.abs(self.spectrum_raw)), cmap='gray')
        ax.set_title(self.image_name)

        roi = {}

        def onselect(eclick, erelease):
            x0, y0 = int(eclick.xdata), int(eclick.ydata)
            x1, y1 = int(erelease.xdata), int(erelease.ydata)

            roi['x'] = min(x0, x1)
            roi['y'] = min(y0, y1)
            roi['width'] = abs(x1 - x0)
            roi['height'] = abs(y1 - y0)

            self.roi_center_x = roi['x'] + roi['width'] // 2
            self.roi_center_y = roi['y'] + roi['height'] // 2
            self.roi_width = roi['width']
            self.roi_height = roi['height']

            plt.close()

        toggle_selector = RectangleSelector(
            ax, onselect, interactive=True, useblit=True,
            button=[1], minspanx=5, minspany=5, spancoords='pixels',
            props=dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)
        )

        plt.tight_layout()
        plt.show()

        if roi:
            y0 = roi['y']
            x0 = roi['x']
            y1 = y0 + roi['height']
            x1 = x0 + roi['width']

            mask = np.zeros_like(self.spectrum_raw, dtype=self.spectrum_raw.dtype)
            mask[y0:y1, x0:x1] = self.spectrum_raw[y0:y1, x0:x1]
            return mask
        else:
            print("No ROI selected, using raw spectrum")
            return self.spectrum_raw

    def Get_Spectrum_Give_Values(self):
        x0 = int(self.roi_center_x - self.roi_width // 2)
        y0 = int(self.roi_center_y - self.roi_height // 2)
        x1 = x0 + self.roi_width
        y1 = y0 + self.roi_height

        mask = np.zeros_like(self.spectrum_raw, dtype=self.spectrum_raw.dtype)
        mask[y0:y1, x0:x1] = self.spectrum_raw[y0:y1, x0:x1]
        return mask

    def Get_Spectrum_Auto_Define(self):
        magnitude = np.abs(self.spectrum_raw)
        log_magnitude = np.log1p(magnitude)

        h, w = log_magnitude.shape
        mask = np.ones_like(log_magnitude)
        cx, cy = w // 2, h // 2
        mask[cy - 20:cy + 20, cx - 20:cx + 20] = 0
        masked = log_magnitude * mask

        y_peak, x_peak = np.unravel_index(np.argmax(masked), masked.shape)

        self.roi_center_x = x_peak
        self.roi_center_y = y_peak

        x0 = int(x_peak - self.roi_width // 2)
        y0 = int(y_peak - self.roi_height // 2)
        x1 = x0 + self.roi_width
        y1 = y0 + self.roi_height

        mask = np.zeros_like(self.spectrum_raw, dtype=self.spectrum_raw.dtype)
        mask[y0:y1, x0:x1] = self.spectrum_raw[y0:y1, x0:x1]
        return mask

class Spectrum_LiuJL():
    def __init__(self, hologram, config):

        self.hologram = hologram.hologram

        self.hologram_spectrum0 = np.zeros(self.hologram.shape, dtype="complex")  # 全息图傅里叶变换之后的频谱原图，要进行处理
        self.hologram_spectrum1 = np.zeros(self.hologram.shape, dtype="complex")  # 频谱图进行转换之后的可见频谱图
        self.hologram_spectrum_raw = np.zeros(self.hologram.shape, dtype="complex")  # 不进行处理
        self.image_height = self.hologram_spectrum1.shape[0]
        self.image_width = self.hologram_spectrum1.shape[1]
        self.point1 = None
        self.point2 = None

        self.holo_type = config.file_info['holo_type']

        self.method = config.spectrum['method']
        self.method_list = config.spectrum['method_list']
        #手动输入频谱亮斑截取阈值
        self.threshold = config.spectrum['threshold']
        # 手动输入的需要截取的频谱图的大小，宽度和长度
        self.center_x = config.spectrum['ROI_rectangle']['center_x']
        self.center_y = config.spectrum['ROI_rectangle']['center_y']
        self.rect_width = config.spectrum['ROI_rectangle']['rect_width']
        self.rect_height = config.spectrum['ROI_rectangle']['rect_height']
        #手动输入中心正方形掩膜的大小
        self.center_masklen = config.spectrum['center_mask_radius']

        self.wave_length = config.image_info['wavelength'] * unit_nm
        self.pixel_size = config.image_info['pixel_size'] * unit_um

        self.spectrum_rect  = []
        self.spectrum_angle = []

    def run(self):
        # self.hologram_spectrum = np.fft.fftshift(np.fft.fft2(self.hologram))
        # self.hologram_spectrum_raw = self.hologram_spectrum.copy()
        #
        # return self.hologram_spectrum, self.hologram_spectrum_raw, self.spectrum_rect

        if self.holo_type == 'Inline':
            self.hologram_spectrum = self.get_hologram_spectrum()
        else:
            if self.method == 'Manual_Select':
                self.hologram_spectrum = self.Get_Spectrum_Manual_Select()
            elif self.method == 'Give_Values':
                self.hologram_spectrum = self.Get_Spectrum_Give_Values()
            elif self.method == 'Auto_Define':
                self.hologram_spectrum = self.Get_Spectrum_Auto_Define()
            else:
                pass
            self.spectrum_rect = [self.center_x, self.center_y, self.rect_width, self.rect_height]

        return self.hologram_spectrum, self.hologram_spectrum_raw, self.spectrum_rect

    '''得到全息图的频谱,并将低频部分移到图像中心'''

    def get_hologram_spectrum(self):
        # 原来：np.fft.fft2，CPU，4508×4096 complex128 约 1.8s
        # 现在：cp.fft.fft2，GPU，预计 < 30ms
        holo_gpu = cp.asarray(self.hologram)
        spec_gpu = cp.fft.fftshift(cp.fft.fft2(holo_gpu))
        del holo_gpu

        self.hologram_spectrum0    = spec_gpu          # 保持为 CuPy 数组，后续操作继续在 GPU
        self.hologram_spectrum_raw = spec_gpu.copy()

        # 可视化用的 log 幅值图，转回 CPU 供 cv2 使用
        abs_gpu = cp.abs(spec_gpu)
        log_gpu = cp.log1p(abs_gpu)
        del abs_gpu
        log_cpu = cp.asnumpy(log_gpu)
        del log_gpu

        self.hologram_spectrum1 = cv2.normalize(
            log_cpu, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return self.hologram_spectrum0

    '''手动截取频谱图'''

    def Get_Spectrum_Manual_Select(self):
        self.get_hologram_spectrum()
        cv2.namedWindow('spectrum', 0)  # 创建一个spectrum的窗口
        cv2.imshow('spectrum', self.hologram_spectrum1)  # 在spectrum的窗口中显示频谱图
        cv2.setMouseCallback('spectrum', self.on_mouse)  # 设置鼠标回调函数，对spectrum这个窗口里面的频谱图进行on_mouse的操作
        cv2.waitKey(0)

        return self.hologram_spectrum0

    def on_mouse(self, event, x, y, flags, param):
        img = self.hologram_spectrum1.copy()  #将频谱图进行复制
        if event == cv2.EVENT_LBUTTONDOWN:  #当鼠标按下时
            self.point1 = (x, y)  #记录下第一个点击的位置坐标，self.point需设置为全局变量
            self.center_x = self.point1[0]
            self.center_y = self.point1[1]

            cv2.circle(img, self.point1, 3, (0, 255, 0), 3)  #在point1的点画一个圆进行标记
            cv2.imshow('spectrum', img)  #在与上面相同的spectrum窗口中显示已画圆的图像
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  #当鼠标移动并且左键按下时
            cv2.rectangle(img, self.point1, (x, y), (255, 0, 0), 3)  #实时画出一个左上角为point1，右下角为现在鼠标实时位置的矩形
            cv2.imshow('spectrum', img)  #在与上面相同的spectrum窗口中显示该矩形
        elif event == cv2.EVENT_LBUTTONUP:  #当鼠标松开时
            self.point2 = (x, y)  #记录下鼠标松开的point2的位置坐标
            self.rect_width = abs(self.point1[0]-self.point2[0])
            self.rect_height = abs(self.point1[1]-self.point2[1])

            cv2.rectangle(img, self.point1, self.point2, (0, 0, 255), 3)  #画出一个左上角为point1，右下角为point2的矩形
            cv2.imshow('spectrum', img)  #在与上面相同的spectrum窗口中显示该矩形
            min_x = min(self.point1[0], self.point2[0])  #找到较小的x
            min_y = min(self.point1[1], self.point2[1])  #找到较小的y
            rectan_width = abs(self.point1[0] - self.point2[0])  #算出矩形的宽
            rectan_height = abs(self.point1[1] - self.point2[1])  #算出矩形的长
            img[:, 0:min_x] = 0  #opencv向下为正方向，且第一个变量是y坐标，将矩形左边置0
            img[0:min_y, min_x:min_x + rectan_width] = 0  #将矩形上方置0
            img[:, min_x + rectan_width:self.image_width] = 0  #将矩形右方置0
            img[min_y + rectan_height:self.image_height, min_x:min_x + rectan_width] = 0  #将矩形下方置0
            cv2.imshow('spectrum', img)  #在与上面相同的spectrum窗口中显示该矩形
            self.hologram_spectrum0[:, 0:min_x] = 0  #对原始全息频谱图进行相同的操作
            self.hologram_spectrum0[0:min_y, min_x:min_x + rectan_width] = 0
            self.hologram_spectrum0[:, min_x + rectan_width:self.image_width] = 0
            self.hologram_spectrum0[min_y + rectan_height:self.image_height, min_x:min_x + rectan_width] = 0

            '''找到中心最亮的点，并找到其矩形轮廓'''
            _, binary_image = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)  #找到像素值高于180的置255，低于180的置0
            binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)  #将输出的二值化图像进行高斯模糊？
            contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  #找出每一轮廓的最小矩形
            counter_x = []
            counter_y = []
            counter_w = []
            counter_h = []
            for contour in contours:  #对所有轮廓进行遍历，并记录下其最上角的位置x,y，以及矩形的宽度w,h
                # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                counter_x.append(x)
                counter_y.append(y)
                counter_w.append(w)
                counter_h.append(h)
                # aspect_ratio = float(w) / h
                # chars.append(img[y:y + h, x:x + w])

            '''找到最大宽度的矩形并画出'''
            max_index = counter_w.index(max(counter_w))  #找到宽度最大矩形的索引，在上述其余三个列表中寻找矩形的其他三个参数
            x_center = counter_x[max_index]
            y_center = counter_y[max_index]
            w_center = counter_w[max_index]
            h_center = counter_h[max_index]
            cv2.rectangle(img, (x_center, y_center), (x_center + w_center, y_center + h_center), (0, 255, 0),
                          8)  #画出包含中心亮斑的宽度最大的矩形，注意y正方向向下
            cv2.namedWindow('spectrum_cut', 0)  #新创建一个窗口命名为spectrum_cut
            cv2.imshow('spectrum_cut', img)  #在这个窗口里面显示被截取后的中心有亮斑的频谱图
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            '''频谱移到整幅图的中心'''
            # delta_x = int(0.5*rectan_width+min_x-0.5*img_width)
            # delta_y = int(0.5*rectan_height+min_y-0.5*img_height)
            delta_x = int(0.5 * w_center + x_center - 0.5 * self.image_width)  #计算截取的频谱图的中心x与整幅图像的中心x的差值
            delta_y = int(0.5 * h_center + y_center - 0.5 * self.image_height)  #同理得y
            self.hologram_spectrum0 = np.roll(self.hologram_spectrum0, -delta_x, axis=1)  #将截取的频谱图沿x移到整个图的x中心
            self.hologram_spectrum0 = np.roll(self.hologram_spectrum0, -delta_y, axis=0)  #同理移y
            '''np.roll 会自动处理边界，滚动超出边界的元素会从另一端重新进入数组，避免了边界问题。'''

            '''计算离轴角'''
            # rect_centerx = x_center + 0.5 * w_center
            # rect_centery = y_center + 0.5 * h_center
            # delta_pixelx = abs(rect_centerx - 0.5 * self.image_width)
            # delta_pixely = abs(rect_centery - 0.5 * self.image_height)
            # offangle_x = np.arcsin(self.wave_length * delta_pixelx / (self.image_width * self.pixel_size))
            # offangle_y = np.arcsin(self.wave_length * delta_pixely / (self.image_height * self.pixel_size))
            # offangle = np.arctan(np.sqrt(np.tan(offangle_x) ** 2 + np.tan(offangle_y) ** 2))
            # offangle = np.degrees(offangle)
            # 注释掉这行，因为offangle变量没有被定义
            # print(f"离轴角: {offangle:.2f} 度")

    def Get_Spectrum_Give_Values(self):
        self.get_hologram_spectrum()
        img = self.hologram_spectrum1.copy()

        '''根据画出的最大矩形，决定框选的频谱大小'''
        img[:, 0:self.center_x] = 0  # opencv向下为正方向，且第一个变量是y坐标，将矩形左边置0
        img[0:self.center_y, :] = 0  # 将矩形上方置0
        img[:, self.center_x+ self.rect_width:self.image_width] = 0  # 将矩形右方置0
        img[self.center_y+ self.rect_height:self.image_height, :] = 0  # 将矩形下方置0

        self.hologram_spectrum0[:, 0:self.center_x] = 0  # opencv向下为正方向，且第一个变量是y坐标，将矩形左边置0
        self.hologram_spectrum0[0:self.center_y, :] = 0  # 将矩形上方置0
        self.hologram_spectrum0[:, self.center_x + self.rect_width:self.image_width] = 0  # 将矩形右方置0
        self.hologram_spectrum0[self.center_y + self.rect_height:self.image_height, :] = 0  # 将矩形下方置0
        # cv2.namedWindow('spectrum_cut', 0)  # 新创建一个窗口命名为spectrum_cut
        # cv2.imshow('spectrum_cut', img)  # 在这个窗口里面显示被截取后的中心有亮斑的频谱图
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # img_plt = Image.fromarray(img, 'L')
        # img_plt.show()

        '''频谱移到整幅图的中心'''
        # delta_x = int(0.5*rectan_width+min_x-0.5*img_width)
        # delta_y = int(0.5*rectan_height+min_y-0.5*img_height)
        delta_x = int(0.5 * self.rect_width + self.center_x - 0.5 * self.image_width)  # 计算截取的频谱图的中心x与整幅图像的中心x的差值
        delta_y = int(0.5 * self.rect_height + self.center_y - 0.5 * self.image_height)  # 同理得y
        self.hologram_spectrum0 = np.roll(self.hologram_spectrum0, -delta_x, axis=1)  # 将截取的频谱图沿x移到整个图的x中心
        self.hologram_spectrum0 = np.roll(self.hologram_spectrum0, -delta_y, axis=0)  # 同理移y
        '''np.roll 会自动处理边界，滚动超出边界的元素会从另一端重新进入数组，避免了边界问题。'''

        # '''计算离轴角'''
        # delta_pixelx = abs(0.5 * self.rect_width + self.center_x - 0.5 * self.image_width)
        # delta_pixely = abs(0.5 * self.rect_height + self.center_y - 0.5 * self.image_height)
        # offangle_x = np.arcsin(self.wave_length * delta_pixelx / (self.image_width * self.pixel_size))
        # offangle_y = np.arcsin(self.wave_length * delta_pixely / (self.image_height * self.pixel_size))
        # offangle = np.arctan(np.sqrt(np.tan(offangle_x) ** 2 + np.tan(offangle_y) ** 2))
        # offangle = np.degrees(offangle)
        # print(f"离轴角: {offangle:.2f} 度")

        return self.hologram_spectrum0

    def Get_Spectrum_Auto_Define(self):
        self.get_hologram_spectrum()

        # contour 检测在 CPU 上跑（hologram_spectrum1 已是 numpy uint8，不变）
        img = self.hologram_spectrum1.copy()
        img[int(0.5 * self.image_height - 0.5 * self.center_masklen):int(
            0.5 * self.image_height + 0.5 * self.center_masklen),
        int(0.5 * self.image_width - 0.5 * self.center_masklen):int(
            0.5 * self.image_width + 0.5 * self.center_masklen)] = 0

        _, binary_image = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
        contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_x, counter_y, counter_w, counter_h = [], [], [], []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            counter_x.append(x)
            counter_y.append(y)
            counter_w.append(w)
            counter_h.append(h)

        max_index = counter_w.index(max(counter_w))
        x_center = counter_x[max_index]
        y_center = counter_y[max_index]
        w_center = counter_w[max_index]
        h_center = counter_h[max_index]

        rect_centerx = x_center + 0.5 * w_center
        rect_centery = y_center + 0.5 * h_center
        self.center_x = int(rect_centerx - 0.5 * self.rect_width)
        self.center_y = int(rect_centery - 0.5 * self.rect_height)

        # 置零：保留 [y0:y1, x0:x1] 区域，其余全部置零
        # hologram_spectrum0 是 CuPy 数组，直接在 GPU 上操作
        x0 = int(rect_centerx - 0.5 * self.rect_width)
        x1 = int(rect_centerx + 0.5 * self.rect_width)
        y0 = int(rect_centery - 0.5 * self.rect_height)
        y1 = int(rect_centery + 0.5 * self.rect_height)

        mask_gpu = cp.zeros_like(self.hologram_spectrum0)
        mask_gpu[y0:y1, x0:x1] = self.hologram_spectrum0[y0:y1, x0:x1]
        self.hologram_spectrum0 = mask_gpu
        del mask_gpu

        # roll：cp.roll 在 GPU 上完成，预计 < 10ms
        delta_x = int(rect_centerx - 0.5 * self.image_width)
        delta_y = int(rect_centery - 0.5 * self.image_height)
        self.hologram_spectrum0 = cp.roll(self.hologram_spectrum0, -delta_x, axis=1)
        self.hologram_spectrum0 = cp.roll(self.hologram_spectrum0, -delta_y, axis=0)

        return self.hologram_spectrum0

if __name__ == '__main__':
    print('Utils Spectrum Module', end='\n\n')

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