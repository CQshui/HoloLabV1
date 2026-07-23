import os
import glob
import cv2
import numpy as np


def _imread(path, flags=cv2.IMREAD_GRAYSCALE):
    """cv2.imread 替代，支持中文/Unicode 路径"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception:
        return None
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from module.hologram import Hologram
from module.config import HoloConfig

class OpenImage():
    def __init__(self, hologram, config, mode = 'Single Image'):
        self._hologram      = hologram
        self.config         = config

        self.holo_type      = self.config.file_info.get('holo_type', None)
        self.holo_type_list = self.config.file_info.get('holo_type_list', None)
        self.mode           = mode # self.config.file_info.get('read_mode', None)  # single or multi

        'single'
        self.image_path     = self.config.file_info.get('image_path', None)
        self.image_name     = self.config.file_info.get('image_name', None)
        self.image_url      = os.path.join(self.image_path, self.image_name)
        self._image_loaded  = hologram._image_loaded

        self.hologram_raw   = None
        self.hologram       = None
        self.image_height   = None
        self.image_width    = None

        'multi'
        self.images_path        = self.config.multi_processing['images_path']
        self.images_name_suffix = self.config.multi_processing['images_name_suffix']
        self.images_name_suffix = "*" if self.images_name_suffix == 'any' else self.images_name_suffix

        if 1:
            self.save_action    = config.save_and_load['save_raw_hologram']
            self.creat_sub_dir  = config.save_and_load['creat_sub_dir']

            self.save_path      = config.save_and_load['data_save_path']
            self.image_name     = os.path.splitext(config.file_info['image_name'])[0]

    def run(self):
        # OpenCV的cv2.imread()函数，当路径无效、读取错误时，不会抛出异常，而是静默返回 None，因此这里没用 try-except

        if self.mode =='Single Image':
            '''若路径有误'''
            if not os.path.exists(self.image_url):
                self.status_msg = f"Image not found: {self.image_url}"
                return self.hologram_raw, self.hologram, self._image_loaded, self.status_msg

            self.hologram_raw = _imread(self.image_url, cv2.IMREAD_GRAYSCALE)
            self.hologram     = _imread(self.image_url, cv2.IMREAD_GRAYSCALE)

            '''若格式问题，判断图像是否正确加载'''
            if self.hologram_raw is None:
                self.status_msg = f"Failed to load image (invalid format or corrupted file): {image_url}"
                return self.hologram_raw, self.hologram, self._image_loaded, self.status_msg

            '''图像加载成功'''
            self._image_loaded = True

            '''图像尺寸'''
            self.image_height, self.image_width = self.hologram_raw.shape

            '''判断全息图像类型'''
            find_peaks = self._holo_type_judgement()
            self.holo_type = self.holo_type_list[1] if find_peaks else self.holo_type_list[0]

            self.status_msg = f"Image opened (may be '{self.holo_type}'): {self.image_url}"
            # return self.hologram_raw, self.hologram, self._image_loaded, self.image_height, self.image_width, self.holo_type, self.status_msg

        else:
            if not os.path.exists(self.images_path):
                self.status_msg = f"No Image found in: {self.images_path}"
                return [], self.status_msg, [], []

            search_pattern = os.path.join(self.images_path, f"*{self.images_name_suffix}")
            self.files_urls = sorted(glob.glob(search_pattern))
            self.files_urls = [f for f in self.files_urls if os.path.isfile(f)]

            # 检查是否包含非图片内容
            _extensions             = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
            self.containt_non_image = False
            self.images_urls        = []
            self.non_image_urls     = []

            for f in self.files_urls:
                if os.path.isfile(f):
                    ext = os.path.splitext(f)[1].lower()  # 获取小写扩展名
                    if ext in _extensions:
                        self.images_urls.append(f)
                    else:
                        self.non_image_urls.append(f)

            if self.non_image_urls:
                self.containt_non_image  = True
                self.status_msg          = f"Found Non-image files: {self.non_image_urls}"
            else:
                self.status_msg          = f"{len(self.files_urls)} images found in {self.images_path}"

        self.modify_hologram_and_config()

    def modify_hologram_and_config(self):
        if self.mode == 'Single Image':
            self._hologram.hologram_raw             = self.hologram_raw
            self._hologram.hologram                 = self.hologram
            self._hologram._image_loaded            = self._image_loaded
            self._hologram.status_msg               = self.status_msg

            self.config.file_info['holo_type']      = self.holo_type
            self.config.image_info['pixel_num_x']   = self.image_height
            self.config.image_info['pixel_num_y']   = self.image_width

        else:
            self.config.multi_processing['images_urls']         = self.images_urls
            self.config.multi_processing['non_images_urls']     = self.non_image_urls
            self.config.multi_processing['containt_non_image']  = self.containt_non_image
            self._hologram.status_msg = self.status_msg

    def _holo_type_judgement(self):

        img = self.hologram_raw
        # 计算傅里叶变换
        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)  # 频谱中心化
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift))  # 幅度谱（对数尺度）

        # 分析频谱（检测非中心亮点）
        h, w = img.shape
        center = (h // 2, w // 2)

        # 忽略中心区域（低频分量）
        mask = np.zeros_like(img)
        cv2.circle(mask, center, min(h,w)//5, 1, -1)  # 中心区域设为0
        masked_spectrum = magnitude_spectrum * (1 - mask)

        # 检测是否有明显峰值
        threshold = np.mean(masked_spectrum) + 2 * np.std(masked_spectrum)
        peaks = masked_spectrum > threshold

        # 可视化（可选）
        if 0:
            plt.figure(figsize=(8, 3))
            plt.subplot(141), plt.imshow(img, cmap='gray')
            plt.title('Input Image'), plt.axis('off')
            plt.subplot(142), plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum'), plt.axis('off')
            plt.subplot(143), plt.imshow(masked_spectrum, cmap='gray')
            plt.title('Masked Spectrum'), plt.axis('off')
            plt.subplot(144), plt.imshow(peaks, cmap='gray')
            plt.title('Peaks'), plt.axis('off')

            plt.tight_layout()
            plt.show()

        return np.any(peaks)

if __name__ == '__main__':

    hologram    = Hologram()
    config      = HoloConfig()

    open_image  = OpenImage(hologram, config)
    hologram_raw, hologram_img, image_loaded, holo_type, status_msg = open_image.run()

    if open_image.hologram_raw is not None and open_image.hologram is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(open_image.hologram_raw, cmap='gray')
        plt.title('Raw Hologram')
        plt.colorbar()

        # Plot processed hologram
        plt.subplot(1, 2, 2)
        plt.imshow(open_image.hologram, cmap='gray')
        plt.title('Processed Hologram')
        plt.colorbar()

        plt.tight_layout()
        plt.show()
    else:
        print(f"Error: {status_msg}")