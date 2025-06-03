from module.hologram import Hologram
from utils.mock_data import MockData
from common.constants import unit_cm, unit_mm, unit_um, unit_nm

import os
import cv2
import numpy as np

'''
1、输入：全息图：self.hologram
2、输出：预处理后的全息图：self.hologram_pro

- 上述这些参数，在Hologram这个类中已经定义，可以直接返回给它的实体
- 先看看Hologram这个类中有哪些量

'''

class PreProcessing_ME():
    def __init__(self, hologram, config):

        self.hologram_raw   = hologram.hologram_raw
        self.hologram       = hologram.hologram
        self.hologram_blank = hologram.hologram_blank

        self.config             = config
        self.blank_image_path   = self.config.pre_process['background_path']
        self.blank_image_name   = self.config.pre_process['background_name']
        self.blank_image_url    = os.path.join(self.blank_image_path, self.blank_image_name)

    def run(self):
        self.Background_Open()

        try:
            self.hologram = self.hologram_raw - self.hologram_blank
            status_msg = 'Pre-processing successfully.'
        except Exception as e:
            status_msg = f'Failed to Pre-processing {e}'

        return self.hologram, status_msg

    def Background_Open(self):
        self.hologram_blank = cv2.imread(self.blank_image_url, cv2.IMREAD_GRAYSCALE)

    def Background_Average(self):
        a = 1

    def Background_Denoise(self):
        a = 1

class PreProcessing():
    def __init__(self,hologram, config):
        self.hologram_raw       = hologram.hologram_raw
        self.hologram           = hologram.hologram

        self.method             = config.pre_process['method']
        self.background_folder  = config.pre_process['background_path']
        self.background_name    = config.pre_process['background_name']
        self.background_url     = os.path.join(self.background_folder, self.background_name)

    def run(self):
        if self.method == 'Subtraction':
            self.hologram = self.Background_Subtraction()
            self.hologram = self.spatial_filter_Gaussian()

        status_msg = 'Pre-processing successfully.'


        return  self.hologram, status_msg

    def Background_Subtraction(self):
        #图片拓展名为bmp和tiff
        image_extensions=['.bmp','.tiff', '.jpg']
        # 判断指定的文件夹是否存在
        if not os.path.isdir(self.background_folder):
            raise ValueError(f'文件夹不存在：{self.background_folder}')
        #遍历文件夹中所有的文件，得到所有文件和文件夹名称
        background_image_list = []
        for image in os.listdir(self.background_folder):
            background_image_full_path = os.path.join(self.background_folder,image)
            #判断所得path是否是文件，若是则被保留
            if os.path.isfile(background_image_full_path):
                ext=os.path.splitext(image)[1].lower()
                #将文件拓展名为bmp、tiff的图片名组合成一个列表并返回
                if ext in image_extensions:
                    background_image_list.append(background_image_full_path)
        #判断是否有背景图片
        image_count=len(background_image_list)
        if image_count==0:
            raise ValueError(f'请检查文件夹中是否有背景图片：{self.background_folder}')
        '''将所有的背景图并进行平均,单张图片也可兼容'''
        #将背景图第一张图片以灰度图的形式读入，并获得其宽、高的像素尺寸，形成一个大小和背景图一样的全0张量
        H,W=cv2.imread(os.path.join(self.background_folder,background_image_list[0]),cv2.IMREAD_GRAYSCALE).shape



        background_sum=np.zeros((H, W), dtype=np.float32)
        #遍历列表中的所有图片名，并以灰度图的形式读入，并将其转化为float32的格式进行加减，再除以背景图片的个数，得到平均背景图
        for item in background_image_list:
            img = cv2.imread(os.path.join(self.background_folder,item) , cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)
            background_sum += img
        background_average = background_sum/image_count
        #将平均背景图转化回uint8格式
        background_average = cv2.normalize(background_average,None,0,255,cv2.NORM_MINMAX)
        background_average = background_average.astype(np.uint8)
        background_average =  cv2.imread(os.path.join(self.background_folder,background_image_list[0]),cv2.IMREAD_GRAYSCALE)
        #保证输入图像与背景图像尺寸一致
        if self.hologram_raw.shape != background_average.shape:
            raise ValueError("输入图像与背景模型尺寸不匹配！")

        #将全息图减去背景图得到hologram_pro
        self.hologram = cv2.absdiff(self.hologram_raw,background_average)

        return self.hologram
    def Background_Average(self):
        a = 1

    '''中值滤波,核大小需为奇数（通常3或5）'''
    def spatial_filter_median(self):
        self.hologram = cv2.medianBlur(self.hologram, 3)

    '''高斯滤波'''
    def spatial_filter_Gaussian(self, kernel_size=5):
        self.hologram = cv2.GaussianBlur(self.hologram, (kernel_size, kernel_size), 0)
        return self.hologram

    '''混合降噪，先中值去椒盐噪声，再高斯平滑'''
    def spatial_fliter_mix(self):
        self.hologram = cv2.GaussianBlur(cv2.medianBlur(self.hologram,3), (5,5), 0)
        return self.hologram

    '''边缘增强'''
    def smart_edge_enhance(self, ksize=3, alpha=0.3):
        lap = cv2.Laplacian(self.hologram, cv2.CV_16S, ksize)
        lap_abs = cv2.convertScaleAbs(lap)
        # 自适应增强
        enhanced = cv2.addWeighted(self.hologram, 1 + alpha, lap_abs, -alpha, 0)
        # 后处理
        _, self.hologram = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.hologram


if __name__ == '__main__':
    print('Utils PreProcessing Module', end='\n\n')

    '''预定一个全息，并加载模拟数据，用于测试'''
    hologram = MockData()
    print(f'- Hologram Data loaded successfully: {hologram.data_path}')